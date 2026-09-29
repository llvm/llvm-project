//===-- llvm/unittests/Target/TargetMachineOptionsTest.cpp ----------
//-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains unit tests for the opaque structure describing options
/// for TargetMachine creation via the C API.
///
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm-c/TargetMachine.h"
#include "llvm/Config/llvm-config.h"
#include "gtest/gtest.h"

namespace llvm {

TEST(TargetMachineCTest, TargetMachineOptions) {
  auto *Options = LLVMCreateTargetMachineOptions();

  LLVMTargetMachineOptionsSetCPU(Options, "cortex-a53");
  LLVMTargetMachineOptionsSetFeatures(Options, "+neon");
  LLVMTargetMachineOptionsSetABI(Options, "aapcs");
  LLVMTargetMachineOptionsSetCodeGenOptLevel(Options, LLVMCodeGenLevelNone);
  LLVMTargetMachineOptionsSetRelocMode(Options, LLVMRelocStatic);
  LLVMTargetMachineOptionsSetCodeModel(Options, LLVMCodeModelKernel);

  LLVMDisposeTargetMachineOptions(Options);
}

TEST(TargetMachineCTest, TargetMachineCreation) {
  LLVMInitializeAllTargets();
  LLVMInitializeAllTargetInfos();
  LLVMInitializeAllTargetMCs();

  // Get the default target to keep the test as generic as possible. This may
  // not be a target for which we can generate code; in that case we give up.

  auto *Triple = LLVMGetDefaultTargetTriple();
  if (strlen(Triple) == 0) {
    LLVMDisposeMessage(Triple);
    GTEST_SKIP();
  }

  LLVMTargetRef Target = nullptr;
  char *Error = nullptr;
  if (LLVMGetTargetFromTriple(Triple, &Target, &Error))
    FAIL() << "Failed to create target from default triple (" << Triple
           << "): " << Error;

  ASSERT_NE(Target, nullptr);

  if (!LLVMTargetHasTargetMachine(Target))
    GTEST_SKIP() << "Default target doesn't support code generation";

  // We don't know which target we're creating a machine for, so don't set any
  // non-default options; they might cause fatal errors.

  auto *Options = LLVMCreateTargetMachineOptions();
  auto *TM = LLVMCreateTargetMachineWithOptions(Target, Triple, Options);
  ASSERT_NE(TM, nullptr);

  LLVMDisposeMessage(Triple);
  LLVMDisposeTargetMachineOptions(Options);
  LLVMDisposeTargetMachine(TM);
}

} // namespace llvm
