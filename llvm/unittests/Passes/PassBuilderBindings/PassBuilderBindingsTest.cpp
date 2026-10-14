//===- unittests/Passes/PassBuilderBindingsTest.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm-c/Target.h"
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
      LLVMDisposeMessage(Err);
      LLVMDisposeMessage(Triple);
      GTEST_SKIP() << "Default target is not built";
    }
    TM = LLVMCreateTargetMachine(Target, Triple, "generic", "",
                                 LLVMCodeGenLevelDefault, LLVMRelocDefault,
                                 LLVMCodeModelDefault);
    LLVMDisposeMessage(Triple);
    Context = LLVMContextCreate();
    Module = LLVMModuleCreateWithNameInContext("test", Context);
    LLVMTypeRef FT =
        LLVMFunctionType(LLVMVoidTypeInContext(Context), nullptr, 0, 0);
    Function = LLVMAddFunction(Module, "test", FT);
  }

  void TearDown() override {
    // SetUp skips when the default triple has no target, leaving these null.
    if (!TM)
      return;
    LLVMDisposeTargetMachine(TM);
    LLVMDisposeModule(Module);
    LLVMContextDispose(Context);
  }

public:
  LLVMTargetMachineRef TM = nullptr;
  LLVMModuleRef Module = nullptr;
  LLVMValueRef Function = nullptr;
  LLVMContextRef Context = nullptr;
};

TEST_F(PassBuilderCTest, Basic) {
  LLVMPassBuilderOptionsRef Options = LLVMCreatePassBuilderOptions();
  LLVMPassBuilderOptionsSetLoopUnrolling(Options, 1);
  LLVMPassBuilderOptionsSetVerifyEach(Options, 1);
  LLVMPassBuilderOptionsSetDebugLogging(Options, 0);
  LLVMPassBuilderOptionsSetAAPipeline(Options, "basic-aa");
  if (LLVMErrorRef E = LLVMRunPasses(Module, "default<O2>", TM, Options)) {
    char *Msg = LLVMGetErrorMessage(E);
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

TEST_F(PassBuilderCTest, Function) {
  LLVMPassBuilderOptionsRef Options = LLVMCreatePassBuilderOptions();
  if (LLVMErrorRef E =
          LLVMRunPassesOnFunction(Function, "no-op-function", TM, Options)) {
    char *Msg = LLVMGetErrorMessage(E);
    LLVMDisposePassBuilderOptions(Options);
    FAIL() << "Failed to run passes on function: " << Msg;
  }
  LLVMDisposePassBuilderOptions(Options);
}
