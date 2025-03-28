//===-- SPIRVCodeGenPreparePass.cpp - preserve masked scatter gather --*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass preserves the intrinsic @llvm.masked.* intrinsics by replacing 
// it with a spv intrinsic
//===----------------------------------------------------------------------===//
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

#include "SPIRV.h"
#include "SPIRVGlobalRegistry.h"
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"

using namespace llvm;

namespace llvm {
void initializeSPIRVCodeGenPreparePass(PassRegistry &);
} // namespace llvm

namespace {
class SPIRVCodeGenPrepare : public ModulePass {

  const SPIRVTargetMachine &TM;

public:
  static char ID;
  SPIRVCodeGenPrepare(const SPIRVTargetMachine &TM) : ModulePass(ID), TM(TM) {
    initializeSPIRVCodeGenPreparePass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override {
    return "SPIRV CodeGen prepare pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ModulePass::getAnalysisUsage(AU);
  }
};

} // namespace

char SPIRVCodeGenPrepare::ID = 0;
INITIALIZE_PASS(SPIRVCodeGenPrepare, "codegen-prepare", "SPIRV codegen prepare",
                false, false)

static bool toSpvOverloadedIntrinsic(IntrinsicInst *II, Intrinsic::ID NewID,
                                     ArrayRef<unsigned> OpNos) {
  Function *F = nullptr;
  if (OpNos.empty()) {
    F = Intrinsic::getOrInsertDeclaration(II->getModule(), NewID);
  } else {
    SmallVector<Type *> Tys;
    for (unsigned OpNo : OpNos) {
      Tys.push_back(II->getOperand(OpNo)->getType());
    }

    F = Intrinsic::getOrInsertDeclaration(II->getModule(), NewID, Tys);
  }
  II->setCalledFunction(F);
  return true;
}

static bool lowerIntrinsicToFunction(IntrinsicInst *Intrinsic,
                                     const SPIRVSubtarget &ST,
                                     SPIRVGlobalRegistry &GR) {
  auto IntrinsicID = Intrinsic->getIntrinsicID();
  if (ST.canUseExtension(
          SPIRV::Extension::Extension::SPV_INTEL_masked_gather_scatter)) {
    switch (IntrinsicID) {
    case Intrinsic::masked_scatter: {
      return toSpvOverloadedIntrinsic(
          Intrinsic, Intrinsic::SPVIntrinsics::spv_masked_scatter,
          {0, 1});
    } break;

    case Intrinsic::masked_gather: {
      VectorType* Vty = dyn_cast<VectorType>(Intrinsic -> getOperand(0) -> getType());
      PointerType* PTy = dyn_cast<PointerType>(Vty -> getElementType());
      
      VectorType* ResVecType = dyn_cast<VectorType>(Intrinsic -> getType());
      Type *CompType = ResVecType -> getElementType();
      GR.addPointerToBaseTypeMap(PTy, CompType);
      return toSpvOverloadedIntrinsic(
          Intrinsic, Intrinsic::SPVIntrinsics::spv_masked_gather, {3, 0});
    } break;
    default:
      break;
    }
  }
  return false;
}

bool SPIRVCodeGenPrepare::runOnModule(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    const SPIRVSubtarget &STI = TM.getSubtarget<SPIRVSubtarget>(F);
    SPIRVGlobalRegistry &GR = *(STI.getSPIRVGlobalRegistry());
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *II = dyn_cast<IntrinsicInst>(&I))
          Changed |= lowerIntrinsicToFunction(II, STI, GR);
      }
    }
  }
  return Changed;
}

ModulePass *llvm::createSPIRVCodeGenPreparePass(const SPIRVTargetMachine &TM) {
  return new SPIRVCodeGenPrepare(TM);
}
