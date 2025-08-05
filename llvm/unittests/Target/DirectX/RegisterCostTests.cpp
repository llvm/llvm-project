//===- llvm/unittests/Target/DirectX/RegisterCostTests.cpp ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DirectXInstrInfo.h"
#include "DirectXTargetLowering.h"
#include "DirectXTargetMachine.h"
#include "TargetInfo/DirectXTargetInfo.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::dxil;

namespace {
class RegisterCostTests : public testing::Test {
protected:
  DirectXInstrInfo DXInstInfo;
  DirectXRegisterInfo RI;
  DirectXTargetLowering *DL;

  virtual void SetUp() {
    LLVMInitializeDirectXTargetMC();
    Target T = getTheDirectXTarget();
    RegisterTargetMachine<DirectXTargetMachine> X(T);
    Triple TT("dxil-pc-shadermodel6.3-library");
    StringRef CPU = "";
    StringRef FS = "";
    DirectXTargetMachine TM(T, TT, CPU, FS, TargetOptions(), Reloc::Static,
                            CodeModel::Small, CodeGenOptLevel::Default, false);
    LLVMContext Context;
    Function *F =
        Function::Create(FunctionType::get(Type::getVoidTy(Context), false),
                         Function::ExternalLinkage, 0);
    DL = new DirectXTargetLowering(TM, *TM.getSubtargetImpl(*F));
    delete F;
  }
  virtual void TearDown() { delete DL; }
};

TEST_F(RegisterCostTests, TestRepRegClassForVTSet) {
  const TargetRegisterClass *RC = DL->getRepRegClassFor(MVT::i32);
  EXPECT_EQ(&dxil::DXILClassRegClass, RC);
}

TEST_F(RegisterCostTests, TestTrivialCopyCostGetter) {

  const TargetRegisterClass *RC = DXInstInfo.getRegisterInfo().getRegClass(0);
  unsigned Cost = RC->getCopyCost();
  EXPECT_EQ(1u, Cost);

  RC = RI.getRegClass(0);
  Cost = RC->getCopyCost();
  EXPECT_EQ(1u, Cost);
}
} // namespace
