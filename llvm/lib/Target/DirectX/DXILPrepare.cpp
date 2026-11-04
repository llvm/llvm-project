//===- DXILPrepare.cpp - Prepare LLVM Module for DXIL encoding ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains passes and utilities to convert a modern LLVM
/// module into a module compatible with the LLVM 3.7-based DirectX Intermediate
/// Language (DXIL).
//===----------------------------------------------------------------------===//

#include "DXILRootSignature.h"
#include "DXILShaderFlags.h"
#include "DXILSignatureAnalysis.h"
#include "DirectX.h"
#include "DirectXIRPasses/DXILAttributes.h"
#include "DirectXIRPasses/PointerTypeAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/VersionTuple.h"

#define DEBUG_TYPE "dxil-prepare"

using namespace llvm;
using namespace llvm::dxil;

namespace {

static void collectDeadStringAttrs(AttributeMask &DeadAttrs, AttributeSet &&AS,
                                   const StringSet<> &LiveKeys,
                                   bool AllowExperimental) {
  for (auto &Attr : AS) {
    if (!Attr.isStringAttribute())
      continue;
    StringRef Key = Attr.getKindAsString();
    if (LiveKeys.contains(Key))
      continue;
    if (AllowExperimental && Key.starts_with("exp-"))
      continue;
    DeadAttrs.addAttribute(Key);
  }
}

static void removeStringFunctionAttributes(Function &F,
                                           bool AllowExperimental) {
  AttributeList Attrs = F.getAttributes();
  const StringSet<> LiveKeys = {"waveops-include-helper-lanes",
                                "fp32-denorm-mode"};
  // Collect DeadKeys in FnAttrs.
  AttributeMask DeadAttrs;
  collectDeadStringAttrs(DeadAttrs, Attrs.getFnAttrs(), LiveKeys,
                         AllowExperimental);
  collectDeadStringAttrs(DeadAttrs, Attrs.getRetAttrs(), LiveKeys,
                         AllowExperimental);

  F.removeFnAttrs(DeadAttrs);
  F.removeRetAttrs(DeadAttrs);
}

class DXILPrepareModuleLegacy : public ModulePass {
public:
  DXILPrepareModuleLegacy() : ModulePass(ID) {}
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DXILMetadataAnalysisWrapperPass>();

    AU.addPreserved<DXILMetadataAnalysisWrapperPass>();
    AU.addPreserved<DXILResourceWrapperPass>();
    AU.addPreserved<RootSignatureAnalysisWrapper>();
    AU.addPreserved<SignatureAnalysisWrapper>();
    AU.addPreserved<ShaderFlagsAnalysisWrapper>();
  }
  bool runOnModule(Module &M) override;
  static char ID; // Pass identification.
};
char DXILPrepareModuleLegacy::ID = 0;

} // end anonymous namespace

static Value *maybeGenerateBitcast(IRBuilder<> &Builder,
                                   PointerTypeMap &PointerTypes,
                                   Instruction &Inst, Value *Operand,
                                   Type *Ty) {
  // Omit bitcasts if the incoming value matches the instruction type.
  auto It = PointerTypes.find(Operand);
  if (It != PointerTypes.end()) {
    auto *OpTy = cast<TypedPointerType>(It->second)->getElementType();
    if (OpTy == Ty)
      return nullptr;
  }

  Type *ValTy = Operand->getType();
  // Also omit the bitcast for matching global array types
  if (auto *GlobalVar = dyn_cast<GlobalVariable>(Operand))
    ValTy = GlobalVar->getValueType();

  if (auto *AI = dyn_cast<AllocaInst>(Operand))
    ValTy = AI->getAllocatedType();

  if (auto *ArrTy = dyn_cast<ArrayType>(ValTy)) {
    Type *ElTy = ArrTy->getElementType();
    if (ElTy == Ty)
      return nullptr;
  }

  // finally, drill down GEP instructions until we get the array
  // that is being accessed, and compare element types
  if (ConstantExpr *GEPInstr = dyn_cast<ConstantExpr>(Operand)) {
    while (GEPInstr->getOpcode() == Instruction::GetElementPtr) {
      Value *OpArg = GEPInstr->getOperand(0);
      if (ConstantExpr *NewGEPInstr = dyn_cast<ConstantExpr>(OpArg)) {
        GEPInstr = NewGEPInstr;
        continue;
      }

      if (auto *GlobalVar = dyn_cast<GlobalVariable>(OpArg))
        ValTy = GlobalVar->getValueType();
      if (auto *AI = dyn_cast<AllocaInst>(Operand))
        ValTy = AI->getAllocatedType();
      if (auto *ArrTy = dyn_cast<ArrayType>(ValTy)) {
        Type *ElTy = ArrTy->getElementType();
        if (ElTy == Ty)
          return nullptr;
      }
      break;
    }
  }

  // Insert bitcasts where we are removing the instruction.
  Builder.SetInsertPoint(&Inst);
  // This code only gets hit in opaque-pointer mode, so the type of the
  // pointer doesn't matter.
  PointerType *PtrTy = cast<PointerType>(Operand->getType());
  return Builder.Insert(
      CastInst::Create(Instruction::BitCast, Operand,
                       Builder.getPtrTy(PtrTy->getAddressSpace())));
}

static bool prepareModule(
    Module &M,
    function_ref<dxil::ModuleMetadataInfo &()> GetModuleMetadataInfo) {
  PointerTypeMap PointerTypes = PointerTypeAnalysis::run(M);
  const AttributeMask &AttrMask = getNonDXILAttributeMask();

  const dxil::ModuleMetadataInfo MetadataInfo = GetModuleMetadataInfo();
  VersionTuple ValVer = MetadataInfo.ValidatorVersion;
  bool AllowExperimental = ValVer.getMajor() == 0 && ValVer.getMinor() == 0;

  for (auto &F : M.functions()) {
    F.removeFnAttrs(AttrMask);
    F.removeRetAttrs(AttrMask);
    // Only remove string attributes if we are not skipping validation.
    // This will reserve the experimental attributes when validation version
    // is 0.0 for experiment mode.
    removeStringFunctionAttributes(F, AllowExperimental);
    for (size_t Idx = 0, End = F.arg_size(); Idx < End; ++Idx)
      F.removeParamAttrs(Idx, AttrMask);

    for (auto &BB : F) {
      IRBuilder<> Builder(&BB);
      for (auto &I : make_early_inc_range(BB)) {

        if (auto *CB = dyn_cast<CallBase>(&I)) {
          CB->removeFnAttrs(AttrMask);
          CB->removeRetAttrs(AttrMask);
          for (size_t Idx = 0, End = CB->arg_size(); Idx < End; ++Idx)
            CB->removeParamAttrs(Idx, AttrMask);
          continue;
        }

        // Emtting NoOp bitcast instructions allows the ValueEnumerator to be
        // unmodified as it reserves instruction IDs during contruction.
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (Value *NoOpBitcast = maybeGenerateBitcast(
                  Builder, PointerTypes, I, LI->getPointerOperand(),
                  LI->getType())) {
            LI->replaceAllUsesWith(
                Builder.CreateLoad(LI->getType(), NoOpBitcast));
            LI->eraseFromParent();
          }
          continue;
        }
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (Value *NoOpBitcast = maybeGenerateBitcast(
                  Builder, PointerTypes, I, SI->getPointerOperand(),
                  SI->getValueOperand()->getType())) {

            SI->replaceAllUsesWith(
                Builder.CreateStore(SI->getValueOperand(), NoOpBitcast));
            SI->eraseFromParent();
          }
          continue;
        }
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          if (Value *NoOpBitcast = maybeGenerateBitcast(
                  Builder, PointerTypes, I, GEP->getPointerOperand(),
                  GEP->getSourceElementType()))
            GEP->setOperand(0, NoOpBitcast);
          continue;
        }
        // An atomic on a float allocation exchanges the bit pattern as an
        // integer, so the pointer element type does not match the value
        // type. Typed pointers need a cast to keep the two in agreement.
        if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
          if (Value *NoOpBitcast = maybeGenerateBitcast(
                  Builder, PointerTypes, I, RMW->getPointerOperand(),
                  RMW->getValOperand()->getType()))
            RMW->setOperand(AtomicRMWInst::getPointerOperandIndex(),
                            NoOpBitcast);
          continue;
        }
        if (auto *CmpXchg = dyn_cast<AtomicCmpXchgInst>(&I)) {
          if (Value *NoOpBitcast = maybeGenerateBitcast(
                  Builder, PointerTypes, I, CmpXchg->getPointerOperand(),
                  CmpXchg->getNewValOperand()->getType()))
            CmpXchg->setOperand(AtomicCmpXchgInst::getPointerOperandIndex(),
                                NoOpBitcast);
          continue;
        }
      }
    }
  }

  return true;
}

bool DXILPrepareModuleLegacy::runOnModule(Module &M) {
  return prepareModule(M, [&]() -> dxil::ModuleMetadataInfo & {
    return getAnalysis<DXILMetadataAnalysisWrapperPass>().getModuleMetadata();
  });
}

PreservedAnalyses DXILPrepareModulePass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  bool Changed = prepareModule(M, [&]() -> dxil::ModuleMetadataInfo & {
    return MAM.getResult<DXILMetadataAnalysis>(M);
  });
  if (!Changed)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none()
      .preserve<DXILMetadataAnalysis>()
      .preserve<DXILResourceAnalysis>()
      .preserve<RootSignatureAnalysis>()
      .preserve<SignatureAnalysis>()
      .preserve<ShaderFlagsAnalysis>();
}

INITIALIZE_PASS_BEGIN(DXILPrepareModuleLegacy, DEBUG_TYPE,
                      "DXIL Prepare Module", false, false)
INITIALIZE_PASS_DEPENDENCY(DXILMetadataAnalysisWrapperPass)
INITIALIZE_PASS_END(DXILPrepareModuleLegacy, DEBUG_TYPE, "DXIL Prepare Module",
                    false, false)

ModulePass *llvm::createDXILPrepareModuleLegacyPass() {
  return new DXILPrepareModuleLegacy();
}
