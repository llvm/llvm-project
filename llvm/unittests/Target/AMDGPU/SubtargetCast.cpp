//===- llvm/unittests/Target/AMDGPU/SubtargetCast.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "R600Subtarget.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<TargetMachine> createTM(StringRef TripleStr,
                                               StringRef CPU) {
  Triple TT(TripleStr);
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget(TT, Error);
  if (!T)
    return nullptr;
  return std::unique_ptr<TargetMachine>(T->createTargetMachine(
      TT, CPU, "", TargetOptions(), std::nullopt, std::nullopt));
}

class SubtargetCastTest : public AMDGPUTestBase {
protected:
  void SetUp() override {
    F = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), false),
                         GlobalValue::ExternalLinkage, "f", M);
    GCNTM = createTM("amdgpu-amd-amdhsa", "gfx900");
    R600TM = createTM("r600", "redwood");
    ASSERT_TRUE(GCNTM);
    ASSERT_TRUE(R600TM);
  }

  LLVMContext Ctx;
  Module M{"M", Ctx};
  Function *F = nullptr;
  std::unique_ptr<TargetMachine> GCNTM;
  std::unique_ptr<TargetMachine> R600TM;
};

TEST_F(SubtargetCastTest, Classof) {
  const TargetSubtargetInfo &GCNST = *GCNTM->getSubtargetImpl(*F);
  const TargetSubtargetInfo &R600ST = *R600TM->getSubtargetImpl(*F);
  EXPECT_TRUE(isa<GCNSubtarget>(GCNST));
  EXPECT_FALSE(isa<R600Subtarget>(GCNST));
  EXPECT_TRUE(isa<R600Subtarget>(R600ST));
  EXPECT_FALSE(isa<GCNSubtarget>(R600ST));
}

#if defined(GTEST_HAS_DEATH_TEST) && !defined(NDEBUG)
TEST_F(SubtargetCastTest, GetSubtargetOfWrongClass) {
  EXPECT_DEATH((void)R600TM->getSubtarget<GCNSubtarget>(*F),
               "argument of incompatible type");
  EXPECT_DEATH((void)GCNTM->getSubtarget<R600Subtarget>(*F),
               "argument of incompatible type");
}
#endif
