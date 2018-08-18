//===- PatternMatchTest.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LegalizerHelperTest.h"

namespace {

// Test CTTZ expansion when CTTZ_ZERO_UNDEF is legal or custom,
// in which case it becomes CTTZ_ZERO_UNDEF with select.
TEST_F(LegalizerHelperTest, LowerBitCountingCTTZ0) {
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(
      A, { getActionDefinitionsBuilder(G_CTTZ_ZERO_UNDEF).legalFor({s64}); });
  // Build Instr
  auto MIBCTTZ = B.buildInstr(TargetOpcode::G_CTTZ, LLT::scalar(64), Copies[0]);
  AInfo Info(MF->getSubtarget());
  LegalizerHelper Helper(*MF, Info);
  // Perform Legalization
  ASSERT_TRUE(Helper.lower(*MIBCTTZ, 0, LLT::scalar(64)) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[CZU:%[0-9]+]]:_(s64) = G_CTTZ_ZERO_UNDEF %0
  CHECK: [[ZERO:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
  CHECK: [[SIXTY4:%[0-9]+]]:_(s64) = G_CONSTANT i64 64
  CHECK: [[CMP:%[0-9]+]]:_(s1) = G_ICMP intpred(eq), %0:_(s64), [[ZERO]]
  CHECK: [[SEL:%[0-9]+]]:_(s64) = G_SELECT [[CMP]]:_(s1), [[SIXTY4]]:_, [[CZU]]
  )";

  // Check
  ASSERT_TRUE(CheckMachineFunction(*MF, CheckStr));
}

// CTTZ expansion in terms of CTLZ
TEST_F(LegalizerHelperTest, LowerBitCountingCTTZ1) {
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A,
                      { getActionDefinitionsBuilder(G_CTLZ).legalFor({s64}); });
  // Build Instr
  auto MIBCTTZ = B.buildInstr(TargetOpcode::G_CTTZ, LLT::scalar(64), Copies[0]);
  AInfo Info(MF->getSubtarget());
  LegalizerHelper Helper(*MF, Info);
  // Perform Legalization
  ASSERT_TRUE(Helper.lower(*MIBCTTZ, 0, LLT::scalar(64)) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[NEG1:%[0-9]+]]:_(s64) = G_CONSTANT i64 -1
  CHECK: [[NOT:%[0-9]+]]:_(s64) = G_XOR %0:_, [[NEG1]]
  CHECK: [[SUB1:%[0-9]+]]:_(s64) = G_ADD %0:_, [[NEG1]]
  CHECK: [[AND1:%[0-9]+]]:_(s64) = G_AND [[NOT]]:_, [[SUB1]]:_
  CHECK: [[CST64:%[0-9]+]]:_(s64) = G_CONSTANT i64 64
  CHECK: [[CTLZ:%[0-9]+]]:_(s64) = G_CTLZ [[AND1]]:_
  CHECK: G_SUB [[CST64]]:_, [[CTLZ]]:_
  )";

  // Check
  ASSERT_TRUE(CheckMachineFunction(*MF, CheckStr));
}

// CTTZ expansion in terms of CTPOP
TEST_F(LegalizerHelperTest, LowerBitCountingCTTZ2) {
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(
      A, { getActionDefinitionsBuilder(G_CTPOP).legalFor({s64}); });
  // Build
  auto MIBCTTZ = B.buildInstr(TargetOpcode::G_CTTZ, LLT::scalar(64), Copies[0]);
  AInfo Info(MF->getSubtarget());
  LegalizerHelper Helper(*MF, Info);
  ASSERT_TRUE(Helper.lower(*MIBCTTZ, 0, LLT::scalar(64)) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[NEG1:%[0-9]+]]:_(s64) = G_CONSTANT i64 -1
  CHECK: [[NOT:%[0-9]+]]:_(s64) = G_XOR %0:_, [[NEG1]]
  CHECK: [[SUB1:%[0-9]+]]:_(s64) = G_ADD %0:_, [[NEG1]]
  CHECK: [[AND1:%[0-9]+]]:_(s64) = G_AND [[NOT]]:_, [[SUB1]]:_
  CHECK: [[POP:%[0-9]+]]:_(s64) = G_CTPOP [[AND1]]
  )";

  // Check
  ASSERT_TRUE(CheckMachineFunction(*MF, CheckStr));
}

// CTTZ_ZERO_UNDEF expansion in terms of CTTZ
TEST_F(LegalizerHelperTest, LowerBitCountingCTTZ3) {
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A,
                      { getActionDefinitionsBuilder(G_CTTZ).legalFor({s64}); });
  // Build
  auto MIBCTTZ =
      B.buildInstr(TargetOpcode::G_CTTZ_ZERO_UNDEF, LLT::scalar(64), Copies[0]);
  AInfo Info(MF->getSubtarget());
  LegalizerHelper Helper(*MF, Info);
  ASSERT_TRUE(Helper.lower(*MIBCTTZ, 0, LLT::scalar(64)) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: CTTZ
  )";

  // Check
  ASSERT_TRUE(CheckMachineFunction(*MF, CheckStr));
}

// CTLZ expansion in terms of CTLZ_ZERO_UNDEF
TEST_F(LegalizerHelperTest, LowerBitCountingCTLZ0) {
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(
      A, { getActionDefinitionsBuilder(G_CTLZ_ZERO_UNDEF).legalFor({s64}); });
  // Build
  auto MIBCTLZ = B.buildInstr(TargetOpcode::G_CTLZ, LLT::scalar(64), Copies[0]);
  AInfo Info(MF->getSubtarget());
  LegalizerHelper Helper(*MF, Info);
  ASSERT_TRUE(Helper.lower(*MIBCTLZ, 0, LLT::scalar(64)) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[CZU:%[0-9]+]]:_(s64) = G_CTLZ_ZERO_UNDEF %0
  CHECK: [[ZERO:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
  CHECK: [[SIXTY4:%[0-9]+]]:_(s64) = G_CONSTANT i64 64
  CHECK: [[CMP:%[0-9]+]]:_(s1) = G_ICMP intpred(eq), %0:_(s64), [[ZERO]]
  CHECK: [[SEL:%[0-9]+]]:_(s64) = G_SELECT [[CMP]]:_(s1), [[SIXTY4]]:_, [[CZU]]
  )";

  // Check
  ASSERT_TRUE(CheckMachineFunction(*MF, CheckStr));
}

// CTLZ expansion
TEST_F(LegalizerHelperTest, LowerBitCountingCTLZ1) {
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A,
                      { getActionDefinitionsBuilder(G_CTPOP).legalFor({s8}); });
  // Build
  // Trunc it to s8.
  LLT s8{LLT::scalar(8)};
  auto MIBTrunc = B.buildTrunc(s8, Copies[0]);
  auto MIBCTLZ = B.buildInstr(TargetOpcode::G_CTLZ, s8, MIBTrunc);
  AInfo Info(MF->getSubtarget());
  LegalizerHelper Helper(*MF, Info);
  ASSERT_TRUE(Helper.lower(*MIBCTLZ, 0, s8) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC
  CHECK: [[Cst1:%[0-9]+]]:_(s8) = G_CONSTANT i8 1
  CHECK: [[Sh1:%[0-9]+]]:_(s8) = G_LSHR [[Trunc]]:_, [[Cst1]]:_
  CHECK: [[Or1:%[0-9]+]]:_(s8) = G_OR [[Trunc]]:_, [[Sh1]]:_
  CHECK: [[Cst2:%[0-9]+]]:_(s8) = G_CONSTANT i8 2
  CHECK: [[Sh2:%[0-9]+]]:_(s8) = G_LSHR [[Or1]]:_, [[Cst2]]:_
  CHECK: [[Or2:%[0-9]+]]:_(s8) = G_OR [[Or1]]:_, [[Sh2]]:_
  CHECK: [[Cst4:%[0-9]+]]:_(s8) = G_CONSTANT i8 4
  CHECK: [[Sh4:%[0-9]+]]:_(s8) = G_LSHR [[Or2]]:_, [[Cst4]]:_
  CHECK: [[Or4:%[0-9]+]]:_(s8) = G_OR [[Or2]]:_, [[Sh4]]:_
  CHECK: [[CTPOP:%[0-9]+]]:_(s8) = G_CTPOP [[Or4]]:_
  CHECK: [[Len:%[0-9]+]]:_(s8) = G_CONSTANT i8 8
  CHECK: [[Sub:%[0-9]+]]:_(s8) = G_SUB [[Len]]:_, [[CTPOP]]:_
  )";

  // Check
  ASSERT_TRUE(CheckMachineFunction(*MF, CheckStr));
}
} // namespace
