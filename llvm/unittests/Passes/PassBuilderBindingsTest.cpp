//===- unittests/Passes/PassBuilderBindingsTest.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm-c/Transforms/PassBuilder.h"
#include "llvm-c/Types.h"
#include "gtest/gtest.h"

using namespace llvm;

class PassBuilderCTest : public testing::Test {
  void SetUp() override {
    LLVMInitializeNativeTarget();
    const char *Triple = LLVMGetDefaultTargetTriple();
    char *Err;
    LLVMTargetRef Target;
    if (LLVMGetTargetFromTriple(Triple, &Target, &Err)) {
      FAIL() << "Failed to create target from default triple: " << Err;
    }
    TM = LLVMCreateTargetMachine(Target, Triple, "generic", "",
                                 LLVMCodeGenLevelDefault, LLVMRelocDefault,
                                 LLVMCodeModelDefault);
    Context = LLVMContextCreate();
    Module = LLVMModuleCreateWithNameInContext("test", Context);
  }

  void TearDown() override {
    LLVMDisposeTargetMachine(TM);
    LLVMDisposeModule(Module);
    LLVMContextDispose(Context);
  }

public:
  LLVMTargetMachineRef TM;
  LLVMModuleRef Module;
  LLVMContextRef Context;
};

TEST_F(PassBuilderCTest, Basic) {
  LLVMPassBuilderOptionsRef Options = LLVMCreatePassBuilderOptions();
  LLVMPassBuilderOptionsSetLoopUnrolling(Options, 1);
  LLVMPassBuilderOptionsSetVerifyEach(Options, 1);
  LLVMPassBuilderOptionsSetDebugLogging(Options, 0);
  if (LLVMErrorRef E = LLVMRunPasses(TM, Module, Options, "default<O2>")) {
    char *Msg = LLVMGetErrorMessage(E);
    LLVMConsumeError(E);
    LLVMDisposePassBuilderOptions(Options);
    FAIL() << "Failed to run passes: " << Msg;
  }
  LLVMDisposePassBuilderOptions(Options);
}

TEST_F(PassBuilderCTest, InvalidPassIsError) {
  LLVMPassBuilderOptionsRef Options = LLVMCreatePassBuilderOptions();
  LLVMErrorRef E1 = LLVMRunPasses(TM, Module, Options, "");
  LLVMErrorRef E2 = LLVMRunPasses(TM, Module, Options, "does-not-exist-pass");
  ASSERT_TRUE(E1);
  ASSERT_TRUE(E2);
  LLVMConsumeError(E1);
  LLVMConsumeError(E2);
  LLVMDisposePassBuilderOptions(Options);
}
