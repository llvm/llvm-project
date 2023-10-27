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
#include <string.h>

using namespace llvm;

class PassBuilderCTest : public testing::Test {
  void SetUp() override {
    char *Triple = LLVMGetDefaultTargetTriple();
    if (strlen(Triple) == 0) {
      GTEST_SKIP();
      LLVMDisposeMessage(Triple);
      return;
    }
    LLVMInitializeAllTargetInfos();
    char *Err;
    LLVMTargetRef Target;
    if (LLVMGetTargetFromTriple(Triple, &Target, &Err)) {
      FAIL() << "Failed to create target from default triple (" << Triple
             << "): " << Err;
    }
    TM = LLVMCreateTargetMachine(Target, Triple, "generic", "",
                                 LLVMCodeGenLevelDefault, LLVMRelocDefault,
                                 LLVMCodeModelDefault);
    LLVMDisposeMessage(Triple);
    Context = LLVMContextCreate();
    Module = LLVMModuleCreateWithNameInContext("test", Context);
  }

  void TearDown() override {
    char *Triple = LLVMGetDefaultTargetTriple();
    if (strlen(Triple) == 0) {
      LLVMDisposeMessage(Triple);
      return; // Skipped, so nothing to tear down
    }
    LLVMDisposeMessage(Triple);
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
  if (LLVMErrorRef E = LLVMRunPasses(Module, "default<O2>", TM, Options)) {
    char *Msg = LLVMGetErrorMessage(E);
    LLVMConsumeError(E);
    LLVMDisposePassBuilderOptions(Options);
    FAIL() << "Failed to run passes: " << Msg;
  }
  LLVMDisposePassBuilderOptions(Options);
}

TEST_F(PassBuilderCTest, InvalidPassIsError) {
  LLVMPassBuilderOptionsRef Options = LLVMCreatePassBuilderOptions();
  LLVMErrorRef E1 = LLVMRunPasses(Module, "", TM, Options);
  LLVMErrorRef E2 = LLVMRunPasses(Module, "does-not-exist-pass", TM, Options);
  ASSERT_TRUE(E1);
  ASSERT_TRUE(E2);
  LLVMConsumeError(E1);
  LLVMConsumeError(E2);
  LLVMDisposePassBuilderOptions(Options);
}
