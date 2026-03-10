//===- SPIRVConvertMaskedMemIntrinsics.cpp - Convert masked mem intrinsics ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts llvm.masked.gather/scatter to spv.masked.gather/scatter
// to prevent them from being scalarized by the generic scalarization pass.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVTargetMachine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-convert-masked-mem-intrinsics"

namespace {

class SPIRVConvertMaskedMemIntrinsics : public ModulePass {
  const SPIRVTargetMachine *TM = nullptr;

public:
  static char ID;

  SPIRVConvertMaskedMemIntrinsics() : ModulePass(ID) {
    initializeSPIRVConvertMaskedMemIntrinsicsPass(
        *PassRegistry::getPassRegistry());
  }

  SPIRVConvertMaskedMemIntrinsics(const SPIRVTargetMachine *TM)
      : ModulePass(ID), TM(TM) {
    initializeSPIRVConvertMaskedMemIntrinsicsPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;

  StringRef getPassName() const override {
    return "SPIRV convert masked memory intrinsics";
  }

private:
  bool processIntrinsic(IntrinsicInst &I);
};

} // namespace

char SPIRVConvertMaskedMemIntrinsics::ID = 0;

INITIALIZE_PASS(SPIRVConvertMaskedMemIntrinsics,
                "spirv-convert-masked-mem-intrinsics",
                "Convert masked memory intrinsics for SPIR-V", false, false)

bool SPIRVConvertMaskedMemIntrinsics::runOnModule(Module &M) {
  if (!TM)
    return false;

  bool Changed = false;
  SmallVector<IntrinsicInst *, 8> ToProcess;

  for (Function &F : M) {
    if (!F.isIntrinsic())
      continue;
    Intrinsic::ID IID = F.getIntrinsicID();
    if (IID != Intrinsic::masked_gather && IID != Intrinsic::masked_scatter)
      continue;

    for (User *U : F.users()) {
      if (auto *II = dyn_cast<IntrinsicInst>(U))
        ToProcess.push_back(II);
    }
  }

  for (IntrinsicInst *II : ToProcess)
    Changed |= processIntrinsic(*II);

  return Changed;
}

bool SPIRVConvertMaskedMemIntrinsics::processIntrinsic(IntrinsicInst &I) {
  if (I.getIntrinsicID() == Intrinsic::masked_gather) {
    IRBuilder<> B(I.getParent());
    B.SetInsertPoint(&I);

    Value *Ptrs = I.getArgOperand(0);
    Value *Mask = I.getArgOperand(1);
    Value *Passthru = I.getArgOperand(2);

    // Alignment is stored as a parameter attribute, not as a regular parameter
    uint32_t Alignment = I.getParamAlign(0).valueOrOne().value();

    SmallVector<Value *, 4> Args = {Ptrs, B.getInt32(Alignment), Mask,
                                    Passthru};
    SmallVector<Type *, 4> Types = {I.getType(), Ptrs->getType(),
                                    Mask->getType(), Passthru->getType()};

    auto *NewI = B.CreateIntrinsic(Intrinsic::spv_masked_gather, Types, Args);
    I.replaceAllUsesWith(NewI);
    I.eraseFromParent();
    return true;
  }

  if (I.getIntrinsicID() == Intrinsic::masked_scatter) {
    IRBuilder<> B(I.getParent());
    B.SetInsertPoint(&I);

    Value *Values = I.getArgOperand(0);
    Value *Ptrs = I.getArgOperand(1);
    Value *Mask = I.getArgOperand(2);

    // Alignment is stored as a parameter attribute on the ptrs parameter (arg
    // 1)
    uint32_t Alignment = I.getParamAlign(1).valueOrOne().value();

    SmallVector<Value *, 4> Args = {Values, Ptrs, B.getInt32(Alignment), Mask};
    SmallVector<Type *, 3> Types = {Values->getType(), Ptrs->getType(),
                                    Mask->getType()};

    B.CreateIntrinsic(Intrinsic::spv_masked_scatter, Types, Args);
    I.eraseFromParent();
    return true;
  }

  return false;
}

ModulePass *
llvm::createSPIRVConvertMaskedMemIntrinsicsPass(const SPIRVTargetMachine *TM) {
  return new SPIRVConvertMaskedMemIntrinsics(TM);
}
