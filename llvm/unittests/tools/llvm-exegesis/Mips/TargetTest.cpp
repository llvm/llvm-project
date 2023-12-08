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
#include "TestBase.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace exegesis {
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

Matcher<MCInst> IsLoadLow16BitImm(unsigned Reg, int64_t Value, bool IsGPR32) {
  const unsigned ZeroReg = IsGPR32 ? Mips::ZERO : Mips::ZERO_64;
  const unsigned ORi = IsGPR32 ? Mips::ORi : Mips::ORi64;
  return AllOf(OpcodeIs(ORi),
               ElementsAre(IsReg(Reg), IsReg(ZeroReg), IsImm(Value)));
}

Matcher<MCInst> IsLoadHigh16BitImm(unsigned Reg, int64_t Value, bool IsGPR32) {
  const unsigned LUi = IsGPR32 ? Mips::LUi : Mips::LUi64;
  return AllOf(OpcodeIs(LUi), ElementsAre(IsReg(Reg), IsImm(Value)));
}

Matcher<MCInst> IsShift(unsigned Reg, uint16_t Amount, bool IsGPR32) {
  const unsigned SLL = IsGPR32 ? Mips::SLL : Mips::SLL64_64;
  return AllOf(OpcodeIs(SLL),
               ElementsAre(IsReg(Reg), IsReg(Reg), IsImm(Amount)));
}

class MipsTargetTest : public MipsTestBase {
protected:
  std::vector<MCInst> setRegTo(unsigned Reg, const APInt &Value) {
    return State.getExegesisTarget().setRegTo(State.getSubtargetInfo(), Reg,
                                              Value);
  }
};

TEST_F(MipsTargetTest, SetGPR32RegTo16BitValue) {
  const uint16_t Value = 0xFFFFU;
  const unsigned Reg = Mips::T0;
  EXPECT_THAT(setRegTo(Reg, APInt(16, Value)),
              ElementsAre(IsLoadLow16BitImm(Reg, Value, true)));
}

TEST_F(MipsTargetTest, SetGPR64RegTo16BitValue) {
  const uint16_t Value = 0xFFFFU;
  const unsigned Reg = Mips::T0_64;
  EXPECT_THAT(setRegTo(Reg, APInt(16, Value)),
              ElementsAre(IsLoadLow16BitImm(Reg, Value, false)));
}

TEST_F(MipsTargetTest, SetGPR32RegTo32BitValue) {
  const uint32_t Value0 = 0xFFFF0000UL;
  const unsigned Reg0 = Mips::T0;
  EXPECT_THAT(setRegTo(Reg0, APInt(32, Value0)),
              ElementsAre(IsLoadHigh16BitImm(Reg0, 0xFFFFU, true)));
  const uint32_t Value1 = 0xFFFFFFFFUL;
  const unsigned Reg1 = Mips::T1;
  EXPECT_THAT(setRegTo(Reg1, APInt(32, Value1)),
              ElementsAre(IsLoadHigh16BitImm(Reg1, 0xFFFFU, true),
                          IsLoadLow16BitImm(Reg1, 0xFFFFU, true)));
}

TEST_F(MipsTargetTest, SetGPR64RegTo32BitValue) {
  const uint32_t Value0 = 0x7FFF0000UL;
  const unsigned Reg0 = Mips::T0_64;
  EXPECT_THAT(setRegTo(Reg0, APInt(32, Value0)),
              ElementsAre(IsLoadHigh16BitImm(Reg0, 0x7FFFU, false)));
  const uint32_t Value1 = 0x7FFFFFFFUL;
  const unsigned Reg1 = Mips::T1_64;
  EXPECT_THAT(setRegTo(Reg1, APInt(32, Value1)),
              ElementsAre(IsLoadHigh16BitImm(Reg1, 0x7FFFU, false),
                          IsLoadLow16BitImm(Reg1, 0xFFFFU, false)));
  const uint32_t Value2 = 0xFFFF0000UL;
  const unsigned Reg2 = Mips::T2_64;
  EXPECT_THAT(setRegTo(Reg2, APInt(32, Value2)),
              ElementsAre(IsLoadLow16BitImm(Reg2, 0xFFFFU, false),
                          IsShift(Reg2, 16, false)));
  const uint32_t Value3 = 0xFFFFFFFFUL;
  const unsigned Reg3 = Mips::T3_64;
  EXPECT_THAT(setRegTo(Reg3, APInt(32, Value3)),
              ElementsAre(IsLoadLow16BitImm(Reg3, 0xFFFFU, false),
                          IsShift(Reg3, 16, false),
                          IsLoadLow16BitImm(Reg3, 0xFFFFU, false)));
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
