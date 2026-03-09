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
#include "SPIRVSubtarget.h"
#include "SPIRVTargetMachine.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "spirv-convert-masked-mem-intrinsics"

namespace {

class SPIRVConvertMaskedMemIntrinsics
    : public FunctionPass,
      public InstVisitor<SPIRVConvertMaskedMemIntrinsics> {
  const SPIRVTargetMachine *TM = nullptr;

public:
  static char ID;

  SPIRVConvertMaskedMemIntrinsics() : FunctionPass(ID) {
    initializeSPIRVConvertMaskedMemIntrinsicsPass(
        *PassRegistry::getPassRegistry());
  }

  SPIRVConvertMaskedMemIntrinsics(const SPIRVTargetMachine *TM)
      : FunctionPass(ID), TM(TM) {
    initializeSPIRVConvertMaskedMemIntrinsicsPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;
  void visitIntrinsicInst(IntrinsicInst &I);

  StringRef getPassName() const override {
    return "SPIRV convert masked memory intrinsics";
  }

private:
  SmallVector<Instruction *, 4> ToErase;
};

} // namespace

char SPIRVConvertMaskedMemIntrinsics::ID = 0;

INITIALIZE_PASS(SPIRVConvertMaskedMemIntrinsics,
                "spirv-convert-masked-mem-intrinsics",
                "Convert masked memory intrinsics for SPIR-V", false, false)

bool SPIRVConvertMaskedMemIntrinsics::runOnFunction(Function &F) {
  if (!TM)
    return false;

  ToErase.clear();
  visit(F);

  for (Instruction *I : ToErase)
    I->eraseFromParent();

  return !ToErase.empty();
}

void SPIRVConvertMaskedMemIntrinsics::visitIntrinsicInst(IntrinsicInst &I) {
  if (I.getIntrinsicID() == Intrinsic::masked_gather) {
    const SPIRVSubtarget &ST =
        TM->getSubtarget<SPIRVSubtarget>(*I.getFunction());
    if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_masked_gather_scatter))
      report_fatal_error(
          "llvm.masked.gather requires SPV_INTEL_masked_gather_scatter");

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
    ToErase.push_back(&I);
  } else if (I.getIntrinsicID() == Intrinsic::masked_scatter) {
    const SPIRVSubtarget &ST =
        TM->getSubtarget<SPIRVSubtarget>(*I.getFunction());
    if (!ST.canUseExtension(SPIRV::Extension::SPV_INTEL_masked_gather_scatter))
      report_fatal_error(
          "llvm.masked.scatter requires SPV_INTEL_masked_gather_scatter");

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
    ToErase.push_back(&I);
  }
}

FunctionPass *
llvm::createSPIRVConvertMaskedMemIntrinsicsPass(const SPIRVTargetMachine *TM) {
  return new SPIRVConvertMaskedMemIntrinsics(TM);
}
