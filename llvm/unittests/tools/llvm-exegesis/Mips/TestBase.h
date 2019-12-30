//===-- TestBase.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Test fixture common to all Mips tests.
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TOOLS_LLVMEXEGESIS_MIPS_TESTBASE_H
#define LLVM_UNITTESTS_TOOLS_LLVMEXEGESIS_MIPS_TESTBASE_H

#include "LlvmState.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

void InitializeMipsExegesisTarget();

class MipsTestBase : public ::testing::Test {
protected:
  MipsTestBase() : State("mips-unknown-linux", "mips32") {}

  static void SetUpTestCase() {
    LLVMInitializeMipsTargetInfo();
    LLVMInitializeMipsTargetMC();
    LLVMInitializeMipsTarget();
    InitializeMipsExegesisTarget();
  }

  const LLVMState State;
};

} // namespace exegesis
} // namespace llvm

#endif
