//===-- TargetTest.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Target.h"

#include <cassert>
#include <memory>

#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "TestBase.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

void InitializeRISCVExegesisTarget();

namespace {

using testing::IsEmpty;
using testing::Not;
using testing::NotNull;

class RISCVTargetTest : public RISCVTestBase {
protected:
  std::vector<MCInst> setRegTo(unsigned Reg, const APInt &Value) {
    return State.getExegesisTarget().setRegTo(State.getSubtargetInfo(), Reg,
                                              Value);
  }
};

TEST_F(RISCVTargetTest, SetRegToConstant) {
  const auto Insts = setRegTo(RISCV::X10, APInt());
  EXPECT_THAT(Insts, Not(IsEmpty()));
}

TEST_F(RISCVTargetTest, DefaultPfmCounters) {
  const std::string Expected = "CYCLES";
  EXPECT_EQ(State.getExegesisTarget().getPfmCounters("").CycleCounter,
            Expected);
  EXPECT_EQ(
      State.getExegesisTarget().getPfmCounters("unknown_cpu").CycleCounter,
      Expected);
}

} // namespace
} // namespace exegesis
} // namespace llvm
