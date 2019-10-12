//===-- TargetTest.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Target.h"

#include <cassert>
#include <memory>

#include "MCTargetDesc/MipsMCTargetDesc.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {

void InitializeMipsExegesisTarget();

namespace {

using testing::AllOf;
using testing::ElementsAre;
using testing::Eq;
using testing::Matcher;
using testing::Property;

Matcher<MCOperand> IsImm(int64_t Value) {
  return AllOf(Property(&MCOperand::isImm, Eq(true)),
               Property(&MCOperand::getImm, Eq(Value)));
}

Matcher<MCOperand> IsReg(unsigned Reg) {
  return AllOf(Property(&MCOperand::isReg, Eq(true)),
               Property(&MCOperand::getReg, Eq(Reg)));
}

Matcher<MCInst> OpcodeIs(unsigned Opcode) {
  return Property(&MCInst::getOpcode, Eq(Opcode));
}

Matcher<MCInst> IsLoadLowImm(int64_t Reg, int64_t Value) {
  return AllOf(OpcodeIs(Mips::ORi),
               ElementsAre(IsReg(Reg), IsReg(Mips::ZERO), IsImm(Value)));
}

constexpr const char kTriple[] = "mips-unknown-linux";

class MipsTargetTest : public ::testing::Test {
protected:
  MipsTargetTest() : State(kTriple, "mips32", "") {}

  static void SetUpTestCase() {
    LLVMInitializeMipsTargetInfo();
    LLVMInitializeMipsTarget();
    LLVMInitializeMipsTargetMC();
    InitializeMipsExegesisTarget();
  }

  std::vector<MCInst> setRegTo(unsigned Reg, const APInt &Value) {
    return State.getExegesisTarget().setRegTo(State.getSubtargetInfo(), Reg,
                                              Value);
  }

  LLVMState State;
};

TEST_F(MipsTargetTest, SetRegToConstant) {
  const uint16_t Value = 0xFFFFU;
  const unsigned Reg = Mips::T0;
  EXPECT_THAT(setRegTo(Reg, APInt(16, Value)),
              ElementsAre(IsLoadLowImm(Reg, Value)));
}

TEST_F(MipsTargetTest, DefaultPfmCounters) {
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
