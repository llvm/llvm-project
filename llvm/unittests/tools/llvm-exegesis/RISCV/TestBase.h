//===-- TestBase.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Test fixture common to all RISC-V tests.
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TOOLS_LLVMEXEGESIS_RISCV_TESTBASE_H
#define LLVM_UNITTESTS_TOOLS_LLVMEXEGESIS_RISCV_TESTBASE_H

#include "LlvmState.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

void InitializeRISCVExegesisTarget();

class RISCVTestBase : public ::testing::Test {
protected:
  RISCVTestBase()
      : State(cantFail(
            LLVMState::Create("riscv64-unknown-linux", "generic-rv64"))) {}

  static void SetUpTestCase() {
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTargetMC();
    LLVMInitializeRISCVTarget();
    InitializeRISCVExegesisTarget();
  }

  const LLVMState State;
};

} // namespace exegesis
} // namespace llvm

#endif
