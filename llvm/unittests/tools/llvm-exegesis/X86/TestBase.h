//===-- TestBase.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Test fixture common to all X86 tests.
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TOOLS_LLVMEXEGESIS_X86_TESTBASE_H
#define LLVM_UNITTESTS_TOOLS_LLVMEXEGESIS_X86_TESTBASE_H

#include "LlvmState.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

void InitializeX86ExegesisTarget();

class X86TestBase : public ::testing::Test {
protected:
  X86TestBase()
      : State(cantFail(LLVMState::Create("x86_64-unknown-linux", "haswell"))) {}

  static void SetUpTestCase() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86Target();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();
    InitializeX86ExegesisTarget();
  }

  const LLVMState State;
};

} // namespace exegesis
} // namespace llvm

#endif
