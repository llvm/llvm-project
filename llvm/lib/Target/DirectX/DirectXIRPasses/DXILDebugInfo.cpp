//===--- DXILDebugInfo.cpp - analysis&lowering for Debug info -*- C++ -*- ---=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILDebugInfo.h"
#include "../DirectX.h"
#include "DXILAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "dx-debug-info"

using namespace llvm;
using namespace llvm::dxil;

static bool lowerDXILDebugInfo(Module &M) {
  // Convert debug markers back to dbg.value intrinsics. Record whether any
  // changes were made.
  bool Modified = M.convertFromNewDbgValues();
  DebugInfoFinder DIF;
  DIF.processModule(M);

  {
    Function *DVDecl = nullptr;

    // Logically these should be variables in the
    //  for (BasicBlock &BB : F) loop.
    // They are defined here and cleared at the start of the loop body to avoid
    // the cost of deconstruction and reconstruction.
    DenseMap<DILocalVariable *, std::pair<Instruction *, DbgValueInst *>>
        DbgValues;
    DenseMap<std::pair<DILocalVariable *, DIExpression *>,
             std::pair<Instruction *, DbgValueInst *>>
        DbgValueFragments;
    // Likewise, logically, this should be a variable in the
    // for (Function &F : M) loop.
    DenseSet<DILocalVariable *> DbgVariablesSeen;

    const AttributeMask &AttrMask = getNonDXILAttributeMask();

    for (Function &F : M) {
      F.removeFnAttrs(AttrMask);
      F.removeRetAttrs(AttrMask);
      for (unsigned ArgNo = 0; ArgNo != F.arg_size(); ++ArgNo)
        F.removeParamAttrs(ArgNo, AttrMask);

      bool IsEntryBlock = true;
      DbgVariablesSeen.clear();
      for (BasicBlock &BB : F) {
        Instruction *NextNonDebugInst = nullptr;
        DbgValues.clear();
        DbgValueFragments.clear();
        for (Instruction &I : make_early_inc_range(reverse(BB))) {
          I.eraseMetadataIf([](unsigned KindID, MDNode *) {
            return KindID == LLVMContext::MD_DIAssignID;
          });
          if (!isa<DbgInfoIntrinsic>(I)) {
            NextNonDebugInst = &I;
            continue;
          }
          if (auto *DL = dyn_cast<DbgLabelInst>(&I)) {
            DL->eraseFromParent();
            Modified = true;
            continue;
          }
          // Process both llvm.dbg.value and llvm.dbg.assign here. We convert
          // llvm.dbg.assign to llvm.dbg.value by dropping the last arguments,
          // and remove redundant llvm.dbg.values.
          if (auto *DV = dyn_cast<DbgValueInst>(&I)) {
            // Keep track of the last location where we saw any debug value for
            // a variable.
            DILocalVariable *V = DV->getVariable();
            DIExpression *E = DV->getExpression();
            // If this is already an llvm.dbg.value instruction that we can
            // keep, just do that, otherwise convert it.
            auto *Val = cast<MetadataAsValue>(DV->getArgOperand(0));
            auto *Var = cast<MetadataAsValue>(DV->getArgOperand(1));
            auto *Expr = cast<MetadataAsValue>(DV->getArgOperand(2));
            bool Replace = DV->getIntrinsicID() != Intrinsic::dbg_value;
            if (!isa<ValueAsMetadata>(Val->getMetadata())) {
              // This may be a DIArgList which is not supported in LLVM 3.7. If
              // it is, we cannot record the new value, but we still need to
              // kill any old value. Do this by poison. We do not know the
              // correct type to use here and arbitrarily use i1.
              // This should never be anything other than ValueAsMetadata or
              // DIArgList, but in manually constructed LLVM IR, it can be.
              // Handle this gracefully by also replacing it with poison.
              Val = MetadataAsValue::get(
                  M.getContext(), ConstantAsMetadata::get(PoisonValue::get(
                                      Type::getInt1Ty(M.getContext()))));
              E = DIExpression::get(M.getContext(), {});
              Expr = MetadataAsValue::get(M.getContext(), E);
              Replace = true;
            }
            std::pair<Instruction *, DbgValueInst *> &DbgValue = DbgValues[V];
            std::pair<Instruction *, DbgValueInst *> &DbgValueFragment =
                DbgValueFragments[{V, E}];
            if (DbgValue.second) {
              // If there is a later value of the same fragment at the same
              // location, this value is redundant.
              if (DbgValueFragment.first == NextNonDebugInst) {
                DV->eraseFromParent();
                Modified = true;
                continue;
              }
              // If there is a later identical value of the same fragment at a
              // later point, and there have been no intervening values of
              // different possibly overlapping fragments, that later value is
              // redundant.
              if (DbgValueFragment.second &&
                  DbgValueFragment.second == DbgValue.second &&
                  DbgValueFragment.second->getValue() == DV->getValue()) {
                DbgValue.second->eraseFromParent();
                Modified = true;
              }
            }
            DbgValueInst *NewDV;
            if (Replace) {
              if (!DVDecl) {
                DVDecl =
                    Intrinsic::getOrInsertDeclaration(&M, Intrinsic::dbg_value);
                AttributeMask AM;
                for (Attribute A : DVDecl->getAttributes().getFnAttrs())
                  if (A.isStringAttribute() ||
                      (A.getKindAsEnum() != Attribute::NoUnwind &&
                       A.getKindAsEnum() != Attribute::Memory))
                    AM.addAttribute(A);
                DVDecl->removeFnAttrs(AM);
              }
              NewDV = cast<DbgValueInst>(
                  CallInst::Create(DVDecl, {Val, Var, Expr}, {}, "",
                                   std::next(DV->getIterator())));
              NewDV->setTailCall();
              NewDV->setDebugLoc(DV->getDebugLoc());
              DV->eraseFromParent();
              Modified = true;
            } else {
              NewDV = DV;
            }
            DbgValue = DbgValueFragment = {NextNonDebugInst, NewDV};
            continue;
          }
        }
        // If this is the entry block, if the first value we see for each debug
        // value is undef, it is redundant.
        if (IsEntryBlock) {
          for (Instruction &I : make_early_inc_range(BB)) {
            auto *DV = dyn_cast<DbgValueInst>(&I);
            if (!DV || DbgVariablesSeen.contains(DV->getVariable()))
              continue;
            if (isa<UndefValue>(DV->getValue())) {
              DV->eraseFromParent();
              Modified = true;
              continue;
            }
            DbgVariablesSeen.insert(DV->getVariable());
          }
        }
        IsEntryBlock = false;
      }
    }
  }

  for (DISubprogram *SP : DIF.subprograms()) {
    if (MDTuple *RN = cast_or_null<MDTuple>(SP->getRawRetainedNodes())) {
      SmallVector<Metadata *> MDs(RN->operands());
      MDs.erase(std::remove_if(MDs.begin(), MDs.end(),
                               [](Metadata *M) { return isa<DILabel>(M); }),
                MDs.end());
      SP->replaceRetainedNodes(MDTuple::get(M.getContext(), MDs));
      Modified = true;
    }
  }

  return Modified;
}

PreservedAnalyses DXILDebugInfo::run(Module &M, ModuleAnalysisManager &) {
  return lowerDXILDebugInfo(M) ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
}

namespace {
class DXILDebugInfoLegacy : public ModulePass {
public:
  static char ID;

  DXILDebugInfoLegacy() : ModulePass(ID) {}

  bool runOnModule(Module &M) override { return lowerDXILDebugInfo(M); }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
} // namespace

char DXILDebugInfoLegacy::ID = 0;

INITIALIZE_PASS(DXILDebugInfoLegacy, DEBUG_TYPE, "DXIL Debug Info", false,
                false)

ModulePass *llvm::createDXILDebugInfoLegacyPass() {
  return new DXILDebugInfoLegacy();
}
