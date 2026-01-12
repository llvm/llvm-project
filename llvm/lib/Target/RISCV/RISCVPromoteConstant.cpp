//==- RISCVPromoteConstant.cpp - Promote constant fp to global for RISC-V --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-promote-const"
#define RISCV_PROMOTE_CONSTANT_NAME "RISC-V Promote Constants"

STATISTIC(NumPromoted, "Number of constant literals promoted to globals");
STATISTIC(NumPromotedUses, "Number of uses of promoted literal constants");

namespace {

class RISCVPromoteConstant : public ModulePass {
public:
  static char ID;
  RISCVPromoteConstant() : ModulePass(ID) {}

  StringRef getPassName() const override { return RISCV_PROMOTE_CONSTANT_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
  }

  /// Iterate over the functions and promote the double fp constants that
  /// would otherwise go into the constant pool to a constant array.
  bool runOnModule(Module &M) override {
    if (skipModule(M))
      return false;
    // TargetMachine and Subtarget are needed to query isFPImmlegal.
    const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
    const TargetMachine &TM = TPC.getTM<TargetMachine>();
    bool Changed = false;
    for (Function &F : M) {
      const RISCVSubtarget &ST = TM.getSubtarget<RISCVSubtarget>(F);
      const RISCVTargetLowering *TLI = ST.getTargetLowering();
      Changed |= runOnFunction(F, TLI);
    }
    return Changed;
  }

private:
  bool runOnFunction(Function &F, const RISCVTargetLowering *TLI);
};
} // end anonymous namespace

char RISCVPromoteConstant::ID = 0;

INITIALIZE_PASS(RISCVPromoteConstant, DEBUG_TYPE, RISCV_PROMOTE_CONSTANT_NAME,
                false, false)

ModulePass *llvm::createRISCVPromoteConstantPass() {
  return new RISCVPromoteConstant();
}

bool RISCVPromoteConstant::runOnFunction(Function &F,
                                         const RISCVTargetLowering *TLI) {
  if (F.hasOptNone() || F.hasOptSize())
    return false;

  // Bail out and make no transformation if the target doesn't support
  // doubles, or if we're not targeting RV64 as we currently see some
  // regressions for those targets.
  if (!TLI->isTypeLegal(MVT::f64) || !TLI->isTypeLegal(MVT::i64))
    return false;

  // Collect all unique double constants and their uses in the function. Use
  // MapVector to preserve insertion order.
  MapVector<ConstantFP *, SmallVector<Use *, 8>> ConstUsesMap;

  for (Instruction &I : instructions(F)) {
    for (Use &U : I.operands()) {
      auto *C = dyn_cast<ConstantFP>(U.get());
      if (!C || !C->getType()->isDoubleTy())
        continue;
      // Do not promote if it wouldn't be loaded from the constant pool.
      if (TLI->isFPImmLegal(C->getValueAPF(), MVT::f64,
                            /*ForCodeSize=*/false))
        continue;
      // Do not promote a constant if it is used as an immediate argument
      // for an intrinsic.
      if (auto *II = dyn_cast<IntrinsicInst>(U.getUser())) {
        Function *IntrinsicFunc = II->getFunction();
        unsigned OperandIdx = U.getOperandNo();
        if (IntrinsicFunc && IntrinsicFunc->getAttributes().hasParamAttr(
                                 OperandIdx, Attribute::ImmArg)) {
          LLVM_DEBUG(dbgs() << "Skipping promotion of constant in: " << *II
                            << " because operand " << OperandIdx
                            << " must be an immediate.\n");
          continue;
        }
      }
      // Note: FP args to inline asm would be problematic if we had a
      // constraint that required an immediate floating point operand. At the
      // time of writing LLVM doesn't recognise such a constraint.
      ConstUsesMap[C].push_back(&U);
    }
  }

  int PromotableConstants = ConstUsesMap.size();
  LLVM_DEBUG(dbgs() << "Found " << PromotableConstants
                    << " promotable constants in " << F.getName() << "\n");
  // Bail out if no promotable constants found, or if only one is found.
  if (PromotableConstants < 2) {
    LLVM_DEBUG(dbgs() << "Performing no promotions as insufficient promotable "
                         "constants found\n");
    return false;
  }

  NumPromoted += PromotableConstants;

  // Create a global array containing the promoted constants.
  Module *M = F.getParent();
  Type *DoubleTy = Type::getDoubleTy(M->getContext());

  SmallVector<Constant *, 16> ConstantVector;
  for (auto const &Pair : ConstUsesMap)
    ConstantVector.push_back(Pair.first);

  ArrayType *ArrayTy = ArrayType::get(DoubleTy, ConstantVector.size());
  Constant *GlobalArrayInitializer =
      ConstantArray::get(ArrayTy, ConstantVector);

  auto *GlobalArray = new GlobalVariable(
      *M, ArrayTy,
      /*isConstant=*/true, GlobalValue::InternalLinkage, GlobalArrayInitializer,
      ".promoted_doubles." + F.getName());

  // A cache to hold the loaded value for a given constant within a basic block.
  DenseMap<std::pair<ConstantFP *, BasicBlock *>, Value *> LocalLoads;

  // Replace all uses with the loaded value.
  unsigned Idx = 0;
  for (auto const &Pair : ConstUsesMap) {
    ConstantFP *Const = Pair.first;
    const SmallVector<Use *, 8> &Uses = Pair.second;

    for (Use *U : Uses) {
      Instruction *UserInst = cast<Instruction>(U->getUser());
      BasicBlock *InsertionBB;

      // If the user is a PHI node, we must insert the load in the
      // corresponding predecessor basic block. Otherwise, it's inserted into
      // the same block as the use.
      if (auto *PN = dyn_cast<PHINode>(UserInst))
        InsertionBB = PN->getIncomingBlock(*U);
      else
        InsertionBB = UserInst->getParent();

      if (isa<CatchSwitchInst>(InsertionBB->getTerminator())) {
        LLVM_DEBUG(dbgs() << "Bailing out: catchswitch means thre is no valid "
                             "insertion point.\n");
        return false;
      }

      auto CacheKey = std::make_pair(Const, InsertionBB);
      Value *LoadedVal = nullptr;

      // Re-use a load if it exists in the insertion block.
      if (LocalLoads.count(CacheKey)) {
        LoadedVal = LocalLoads.at(CacheKey);
      } else {
        // Otherwise, create a new GEP and Load at the correct insertion point.
        // It is always safe to insert in the first insertion point in the BB,
        // so do that and let other passes reorder.
        IRBuilder<> Builder(InsertionBB, InsertionBB->getFirstInsertionPt());
        Value *ElementPtr = Builder.CreateConstInBoundsGEP2_64(
            GlobalArray->getValueType(), GlobalArray, 0, Idx, "double.addr");
        LoadedVal = Builder.CreateLoad(DoubleTy, ElementPtr, "double.val");

        // Cache the newly created load for this block.
        LocalLoads[CacheKey] = LoadedVal;
      }

      U->set(LoadedVal);
      ++NumPromotedUses;
    }
    ++Idx;
  }

  return true;
}
