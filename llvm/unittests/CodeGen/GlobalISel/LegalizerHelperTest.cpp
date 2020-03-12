//===- LegalizerHelperTest.cpp
//-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GISelMITest.h"

using namespace LegalizeActions;
using namespace LegalizeMutations;
using namespace LegalityPredicates;

namespace {

class DummyGISelObserver : public GISelChangeObserver {
public:
  void changingInstr(MachineInstr &MI) override {}
  void changedInstr(MachineInstr &MI) override {}
  void createdInstr(MachineInstr &MI) override {}
  void erasingInstr(MachineInstr &MI) override {}
};

// Test CTTZ expansion when CTTZ_ZERO_UNDEF is legal or custom,
// in which case it becomes CTTZ_ZERO_UNDEF with select.
TEST_F(GISelMITest, LowerBitCountingCTTZ0) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTTZ_ZERO_UNDEF).legalFor({{s32, s64}});
  });
  // Build Instr
  auto MIBCTTZ =
      B.buildInstr(TargetOpcode::G_CTTZ, {LLT::scalar(32)}, {Copies[0]});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  // Perform Legalization
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.lower(*MIBCTTZ, 0, LLT::scalar(64)));

  auto CheckStr = R"(
  CHECK: [[CZU:%[0-9]+]]:_(s32) = G_CTTZ_ZERO_UNDEF %0
  CHECK: [[ZERO:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
  CHECK: [[CMP:%[0-9]+]]:_(s1) = G_ICMP intpred(eq), %0:_(s64), [[ZERO]]
  CHECK: [[SIXTY4:%[0-9]+]]:_(s32) = G_CONSTANT i32 64
  CHECK: [[SEL:%[0-9]+]]:_(s32) = G_SELECT [[CMP]]:_(s1), [[SIXTY4]]:_, [[CZU]]
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTTZ expansion in terms of CTLZ
TEST_F(GISelMITest, LowerBitCountingCTTZ1) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTLZ).legalFor({{s64, s64}});
  });
  // Build Instr
  auto MIBCTTZ =
      B.buildInstr(TargetOpcode::G_CTTZ, {LLT::scalar(64)}, {Copies[0]});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  // Perform Legalization
  EXPECT_TRUE(Helper.lower(*MIBCTTZ, 0, LLT::scalar(64)) ==
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
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTLZ scalar narrowing
TEST_F(GISelMITest, NarrowScalarCTLZ) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTLZ).legalFor({{s32, s32}});
  });
  // Build Instr
  auto CTLZ =
      B.buildInstr(TargetOpcode::G_CTLZ, {LLT::scalar(32)}, {Copies[0]});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  // Perform Legalization
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.narrowScalar(*CTLZ, 1, LLT::scalar(32)));

  auto CheckStr = R"(
  CHECK: [[UNMERGE_LO:%[0-9]+]]:_(s32), [[UNMERGE_HI:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES %0:_(s64)
  CHECK: [[ZERO:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
  CHECK: [[CMP:%[0-9]+]]:_(s1) = G_ICMP intpred(eq), [[UNMERGE_HI]]:_(s32), [[ZERO]]:_
  CHECK: [[CTLZ_LO:%[0-9]+]]:_(s32) = G_CTLZ [[UNMERGE_LO]]:_(s32)
  CHECK: [[THIRTYTWO:%[0-9]+]]:_(s32) = G_CONSTANT i32 32
  CHECK: [[ADD:%[0-9]+]]:_(s32) = G_ADD [[CTLZ_LO]]:_, [[THIRTYTWO]]:_
  CHECK: [[CTLZ_HI:%[0-9]+]]:_(s32) = G_CTLZ_ZERO_UNDEF [[UNMERGE_HI]]:_(s32)
  CHECK: %{{[0-9]+}}:_(s32) = G_SELECT [[CMP]]:_(s1), [[ADD]]:_, [[CTLZ_HI]]:_
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTTZ scalar narrowing
TEST_F(GISelMITest, NarrowScalarCTTZ) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTTZ).legalFor({{s32, s64}});
  });
  // Build Instr
  auto CTTZ =
      B.buildInstr(TargetOpcode::G_CTTZ, {LLT::scalar(32)}, {Copies[0]});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  // Perform Legalization
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.narrowScalar(*CTTZ, 1, LLT::scalar(32)));

  auto CheckStr = R"(
  CHECK: [[UNMERGE_LO:%[0-9]+]]:_(s32), [[UNMERGE_HI:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES %0:_(s64)
  CHECK: [[ZERO:%[0-9]+]]:_(s32) = G_CONSTANT i32 0
  CHECK: [[CMP:%[0-9]+]]:_(s1) = G_ICMP intpred(eq), [[UNMERGE_LO]]:_(s32), [[ZERO]]:_
  CHECK: [[CTTZ_HI:%[0-9]+]]:_(s32) = G_CTTZ [[UNMERGE_HI]]:_(s32)
  CHECK: [[THIRTYTWO:%[0-9]+]]:_(s32) = G_CONSTANT i32 32
  CHECK: [[ADD:%[0-9]+]]:_(s32) = G_ADD [[CTTZ_HI]]:_, [[THIRTYTWO]]:_
  CHECK: [[CTTZ_LO:%[0-9]+]]:_(s32) = G_CTTZ_ZERO_UNDEF [[UNMERGE_LO]]:_(s32)
  CHECK: %{{[0-9]+}}:_(s32) = G_SELECT [[CMP]]:_(s1), [[ADD]]:_, [[CTTZ_LO]]:_
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTTZ expansion in terms of CTPOP
TEST_F(GISelMITest, LowerBitCountingCTTZ2) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTPOP).legalFor({{s64, s64}});
  });
  // Build
  auto MIBCTTZ =
      B.buildInstr(TargetOpcode::G_CTTZ, {LLT::scalar(64)}, {Copies[0]});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_TRUE(Helper.lower(*MIBCTTZ, 0, LLT::scalar(64)) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[NEG1:%[0-9]+]]:_(s64) = G_CONSTANT i64 -1
  CHECK: [[NOT:%[0-9]+]]:_(s64) = G_XOR %0:_, [[NEG1]]
  CHECK: [[SUB1:%[0-9]+]]:_(s64) = G_ADD %0:_, [[NEG1]]
  CHECK: [[AND1:%[0-9]+]]:_(s64) = G_AND [[NOT]]:_, [[SUB1]]:_
  CHECK: [[POP:%[0-9]+]]:_(s64) = G_CTPOP [[AND1]]
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTPOP widening.
TEST_F(GISelMITest, WidenBitCountingCTPOP1) {
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
      getActionDefinitionsBuilder(G_CTPOP).legalFor({{s16, s16}});
    });

  // Build
  // Trunc it to s8.
  LLT s8{LLT::scalar(8)};
  LLT s16{LLT::scalar(16)};
  auto MIBTrunc = B.buildTrunc(s8, Copies[0]);
  auto MIBCTPOP = B.buildInstr(TargetOpcode::G_CTPOP, {s16}, {MIBTrunc});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.widenScalar(*MIBCTPOP, 1, s16));

  auto CheckStr = R"(
  CHECK: [[TRUNC:%[0-9]+]]:_(s8) = G_TRUNC %0:_(s64)
  CHECK: [[ZEXT:%[0-9]+]]:_(s16) = G_ZEXT [[TRUNC]]:_(s8)
  CHECK: [[CTPOP:%[0-9]+]]:_(s16) = G_CTPOP [[ZEXT]]
  CHECK: [[COPY:%[0-9]+]]:_(s16) = COPY [[CTPOP]]:_(s16)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// Test a strange case where the result is wider than the source
TEST_F(GISelMITest, WidenBitCountingCTPOP2) {
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
      getActionDefinitionsBuilder(G_CTPOP).legalFor({{s32, s16}});
    });

  // Build
  // Trunc it to s8.
  LLT s8{LLT::scalar(8)};
  LLT s16{LLT::scalar(16)};
  LLT s32{LLT::scalar(32)};
  auto MIBTrunc = B.buildTrunc(s8, Copies[0]);
  auto MIBCTPOP = B.buildInstr(TargetOpcode::G_CTPOP, {s32}, {MIBTrunc});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.widenScalar(*MIBCTPOP, 1, s16));

  auto CheckStr = R"(
  CHECK: [[TRUNC:%[0-9]+]]:_(s8) = G_TRUNC %0:_(s64)
  CHECK: [[ZEXT:%[0-9]+]]:_(s16) = G_ZEXT [[TRUNC]]:_(s8)
  CHECK: [[CTPOP:%[0-9]+]]:_(s16) = G_CTPOP [[ZEXT]]
  CHECK: [[COPY:%[0-9]+]]:_(s32) = G_ZEXT [[CTPOP]]:_(s16)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTTZ_ZERO_UNDEF expansion in terms of CTTZ
TEST_F(GISelMITest, LowerBitCountingCTTZ3) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTTZ).legalFor({{s64, s64}});
  });
  // Build
  auto MIBCTTZ = B.buildInstr(TargetOpcode::G_CTTZ_ZERO_UNDEF,
                              {LLT::scalar(64)}, {Copies[0]});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_TRUE(Helper.lower(*MIBCTTZ, 0, LLT::scalar(64)) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: CTTZ
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTLZ expansion in terms of CTLZ_ZERO_UNDEF
TEST_F(GISelMITest, LowerBitCountingCTLZ0) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTLZ_ZERO_UNDEF).legalFor({{s64, s64}});
  });
  // Build
  auto MIBCTLZ =
      B.buildInstr(TargetOpcode::G_CTLZ, {LLT::scalar(64)}, {Copies[0]});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_TRUE(Helper.lower(*MIBCTLZ, 0, LLT::scalar(64)) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[CZU:%[0-9]+]]:_(s64) = G_CTLZ_ZERO_UNDEF %0
  CHECK: [[ZERO:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
  CHECK: [[CMP:%[0-9]+]]:_(s1) = G_ICMP intpred(eq), %0:_(s64), [[ZERO]]
  CHECK: [[SIXTY4:%[0-9]+]]:_(s64) = G_CONSTANT i64 64
  CHECK: [[SEL:%[0-9]+]]:_(s64) = G_SELECT [[CMP]]:_(s1), [[SIXTY4]]:_, [[CZU]]
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTLZ expansion in terms of CTLZ_ZERO_UNDEF if the latter is a libcall
TEST_F(GISelMITest, LowerBitCountingCTLZLibcall) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTLZ_ZERO_UNDEF).libcallFor({{s32, s64}});
  });
  // Build
  auto MIBCTLZ =
      B.buildInstr(TargetOpcode::G_CTLZ, {LLT::scalar(32)}, {Copies[0]});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.lower(*MIBCTLZ, 0, LLT::scalar(32)));

  auto CheckStr = R"(
  CHECK: [[CZU:%[0-9]+]]:_(s32) = G_CTLZ_ZERO_UNDEF %0
  CHECK: [[ZERO:%[0-9]+]]:_(s64) = G_CONSTANT i64 0
  CHECK: [[CMP:%[0-9]+]]:_(s1) = G_ICMP intpred(eq), %0:_(s64), [[ZERO]]
  CHECK: [[THIRTY2:%[0-9]+]]:_(s32) = G_CONSTANT i32 64
  CHECK: [[SEL:%[0-9]+]]:_(s32) = G_SELECT [[CMP]]:_(s1), [[THIRTY2]]:_, [[CZU]]
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTLZ expansion
TEST_F(GISelMITest, LowerBitCountingCTLZ1) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTPOP).legalFor({{s8, s8}});
  });
  // Build
  // Trunc it to s8.
  LLT s8{LLT::scalar(8)};
  auto MIBTrunc = B.buildTrunc(s8, Copies[0]);
  auto MIBCTLZ = B.buildInstr(TargetOpcode::G_CTLZ, {s8}, {MIBTrunc});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_TRUE(Helper.lower(*MIBCTLZ, 0, s8) ==
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
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTLZ widening.
TEST_F(GISelMITest, WidenBitCountingCTLZ) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTLZ).legalFor({{s16, s16}});
  });
  // Build
  // Trunc it to s8.
  LLT s8{LLT::scalar(8)};
  LLT s16{LLT::scalar(16)};
  auto MIBTrunc = B.buildTrunc(s8, Copies[0]);
  auto MIBCTLZ = B.buildInstr(TargetOpcode::G_CTLZ, {s8}, {MIBTrunc});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_TRUE(Helper.widenScalar(*MIBCTLZ, 1, s16) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC
  CHECK: [[Zext:%[0-9]+]]:_(s16) = G_ZEXT [[Trunc]]
  CHECK: [[Ctlz:%[0-9]+]]:_(s16) = G_CTLZ [[Zext]]
  CHECK: [[Cst8:%[0-9]+]]:_(s16) = G_CONSTANT i16 8
  CHECK: [[Sub:%[0-9]+]]:_(s16) = G_SUB [[Ctlz]]:_, [[Cst8]]:_
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC [[Sub]]
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTLZ_ZERO_UNDEF widening.
TEST_F(GISelMITest, WidenBitCountingCTLZZeroUndef) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTLZ_ZERO_UNDEF).legalFor({{s16, s16}});
  });
  // Build
  // Trunc it to s8.
  LLT s8{LLT::scalar(8)};
  LLT s16{LLT::scalar(16)};
  auto MIBTrunc = B.buildTrunc(s8, Copies[0]);
  auto MIBCTLZ_ZU =
      B.buildInstr(TargetOpcode::G_CTLZ_ZERO_UNDEF, {s8}, {MIBTrunc});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_TRUE(Helper.widenScalar(*MIBCTLZ_ZU, 1, s16) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC
  CHECK: [[Zext:%[0-9]+]]:_(s16) = G_ZEXT [[Trunc]]
  CHECK: [[CtlzZu:%[0-9]+]]:_(s16) = G_CTLZ_ZERO_UNDEF [[Zext]]
  CHECK: [[Cst8:%[0-9]+]]:_(s16) = G_CONSTANT i16 8
  CHECK: [[Sub:%[0-9]+]]:_(s16) = G_SUB [[CtlzZu]]:_, [[Cst8]]:_
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC [[Sub]]
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTPOP widening.
TEST_F(GISelMITest, WidenBitCountingCTPOP) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTPOP).legalFor({{s16, s16}});
  });
  // Build
  // Trunc it to s8.
  LLT s8{LLT::scalar(8)};
  LLT s16{LLT::scalar(16)};
  auto MIBTrunc = B.buildTrunc(s8, Copies[0]);
  auto MIBCTPOP = B.buildInstr(TargetOpcode::G_CTPOP, {s8}, {MIBTrunc});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_TRUE(Helper.widenScalar(*MIBCTPOP, 1, s16) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC
  CHECK: [[Zext:%[0-9]+]]:_(s16) = G_ZEXT [[Trunc]]
  CHECK: [[Ctpop:%[0-9]+]]:_(s16) = G_CTPOP [[Zext]]
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC [[Ctpop]]
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTTZ_ZERO_UNDEF widening.
TEST_F(GISelMITest, WidenBitCountingCTTZ_ZERO_UNDEF) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTTZ_ZERO_UNDEF).legalFor({{s16, s16}});
  });
  // Build
  // Trunc it to s8.
  LLT s8{LLT::scalar(8)};
  LLT s16{LLT::scalar(16)};
  auto MIBTrunc = B.buildTrunc(s8, Copies[0]);
  auto MIBCTTZ_ZERO_UNDEF =
      B.buildInstr(TargetOpcode::G_CTTZ_ZERO_UNDEF, {s8}, {MIBTrunc});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_TRUE(Helper.widenScalar(*MIBCTTZ_ZERO_UNDEF, 1, s16) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC
  CHECK: [[Zext:%[0-9]+]]:_(s16) = G_ZEXT [[Trunc]]
  CHECK: [[CttzZu:%[0-9]+]]:_(s16) = G_CTTZ_ZERO_UNDEF [[Zext]]
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC [[CttzZu]]
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// CTTZ widening.
TEST_F(GISelMITest, WidenBitCountingCTTZ) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTTZ).legalFor({{s16, s16}});
  });
  // Build
  // Trunc it to s8.
  LLT s8{LLT::scalar(8)};
  LLT s16{LLT::scalar(16)};
  auto MIBTrunc = B.buildTrunc(s8, Copies[0]);
  auto MIBCTTZ = B.buildInstr(TargetOpcode::G_CTTZ, {s8}, {MIBTrunc});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_TRUE(Helper.widenScalar(*MIBCTTZ, 1, s16) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC
  CHECK: [[Zext:%[0-9]+]]:_(s16) = G_ZEXT [[Trunc]]
  CHECK: [[Cst:%[0-9]+]]:_(s16) = G_CONSTANT i16 256
  CHECK: [[Or:%[0-9]+]]:_(s16) = G_OR [[Zext]]:_, [[Cst]]
  CHECK: [[Cttz:%[0-9]+]]:_(s16) = G_CTTZ [[Or]]
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC [[Cttz]]
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}
// UADDO widening.
TEST_F(GISelMITest, WidenUADDO) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_ADD).legalFor({{s16, s16}});
  });
  // Build
  // Trunc it to s8.
  LLT s8{LLT::scalar(8)};
  LLT s16{LLT::scalar(16)};
  auto MIBTrunc = B.buildTrunc(s8, Copies[0]);
  unsigned CarryReg = MRI->createGenericVirtualRegister(LLT::scalar(1));
  auto MIBUAddO =
      B.buildInstr(TargetOpcode::G_UADDO, {s8, CarryReg}, {MIBTrunc, MIBTrunc});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_TRUE(Helper.widenScalar(*MIBUAddO, 0, s16) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC
  CHECK: [[LHS:%[0-9]+]]:_(s16) = G_ZEXT [[Trunc]]
  CHECK: [[RHS:%[0-9]+]]:_(s16) = G_ZEXT [[Trunc]]
  CHECK: [[ADD:%[0-9]+]]:_(s16) = G_ADD [[LHS]]:_, [[RHS]]:_
  CHECK: [[CST:%[0-9]+]]:_(s16) = G_CONSTANT i16 255
  CHECK: [[AND:%[0-9]+]]:_(s16) = G_AND [[ADD]]:_, [[CST]]:_
  CHECK: G_ICMP intpred(ne), [[ADD]]:_(s16), [[AND]]:_
  CHECK: G_TRUNC [[ADD]]
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// USUBO widening.
TEST_F(GISelMITest, WidenUSUBO) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_SUB).legalFor({{s16, s16}});
  });
  // Build
  // Trunc it to s8.
  LLT s8{LLT::scalar(8)};
  LLT s16{LLT::scalar(16)};
  auto MIBTrunc = B.buildTrunc(s8, Copies[0]);
  unsigned CarryReg = MRI->createGenericVirtualRegister(LLT::scalar(1));
  auto MIBUSUBO =
      B.buildInstr(TargetOpcode::G_USUBO, {s8, CarryReg}, {MIBTrunc, MIBTrunc});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_TRUE(Helper.widenScalar(*MIBUSUBO, 0, s16) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[Trunc:%[0-9]+]]:_(s8) = G_TRUNC
  CHECK: [[LHS:%[0-9]+]]:_(s16) = G_ZEXT [[Trunc]]
  CHECK: [[RHS:%[0-9]+]]:_(s16) = G_ZEXT [[Trunc]]
  CHECK: [[SUB:%[0-9]+]]:_(s16) = G_SUB [[LHS]]:_, [[RHS]]:_
  CHECK: [[CST:%[0-9]+]]:_(s16) = G_CONSTANT i16 255
  CHECK: [[AND:%[0-9]+]]:_(s16) = G_AND [[SUB]]:_, [[CST]]:_
  CHECK: G_ICMP intpred(ne), [[SUB]]:_(s16), [[AND]]:_
  CHECK: G_TRUNC [[SUB]]
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, FewerElementsAnd) {
  if (!TM)
    return;

  const LLT V2S32 = LLT::vector(2, 32);
  const LLT V5S32 = LLT::vector(5, 32);

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_AND)
      .legalFor({s32});
  });

  auto Op0 = B.buildUndef(V5S32);
  auto Op1 = B.buildUndef(V5S32);
  auto And = B.buildAnd(V5S32, Op0, Op1);

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_TRUE(Helper.fewerElementsVector(*And, 0, V2S32) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[IMP_DEF0:%[0-9]+]]:_(<5 x s32>) = G_IMPLICIT_DEF
  CHECK: [[IMP_DEF1:%[0-9]+]]:_(<5 x s32>) = G_IMPLICIT_DEF
  CHECK: [[IMP_DEF2:%[0-9]+]]:_(<5 x s32>) = G_IMPLICIT_DEF
  CHECK: [[EXTRACT0:%[0-9]+]]:_(<2 x s32>) = G_EXTRACT [[IMP_DEF0]]:_(<5 x s32>), 0
  CHECK: [[EXTRACT1:%[0-9]+]]:_(<2 x s32>) = G_EXTRACT [[IMP_DEF1]]:_(<5 x s32>), 0
  CHECK: [[AND0:%[0-9]+]]:_(<2 x s32>) = G_AND [[EXTRACT0]]:_, [[EXTRACT1]]:_
  CHECK: [[INSERT0:%[0-9]+]]:_(<5 x s32>) = G_INSERT [[IMP_DEF2]]:_, [[AND0]]:_(<2 x s32>), 0

  CHECK: [[EXTRACT2:%[0-9]+]]:_(<2 x s32>) = G_EXTRACT [[IMP_DEF0]]:_(<5 x s32>), 64
  CHECK: [[EXTRACT3:%[0-9]+]]:_(<2 x s32>) = G_EXTRACT [[IMP_DEF1]]:_(<5 x s32>), 64
  CHECK: [[AND1:%[0-9]+]]:_(<2 x s32>) = G_AND [[EXTRACT2]]:_, [[EXTRACT3]]:_
  CHECK: [[INSERT1:%[0-9]+]]:_(<5 x s32>) = G_INSERT [[INSERT0]]:_, [[AND1]]:_(<2 x s32>), 64

  CHECK: [[EXTRACT4:%[0-9]+]]:_(s32) = G_EXTRACT [[IMP_DEF0]]:_(<5 x s32>), 128
  CHECK: [[EXTRACT5:%[0-9]+]]:_(s32) = G_EXTRACT [[IMP_DEF1]]:_(<5 x s32>), 128
  CHECK: [[AND2:%[0-9]+]]:_(s32) = G_AND [[EXTRACT4]]:_, [[EXTRACT5]]:_
  CHECK: [[INSERT2:%[0-9]+]]:_(<5 x s32>) = G_INSERT [[INSERT1]]:_, [[AND2]]:_(s32), 128
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, MoreElementsAnd) {
  if (!TM)
    return;

  LLT s32 = LLT::scalar(32);
  LLT v2s32 = LLT::vector(2, 32);
  LLT v6s32 = LLT::vector(6, 32);

  LegalizerInfo LI;
  LI.getActionDefinitionsBuilder(TargetOpcode::G_AND)
    .legalFor({v6s32})
    .clampMinNumElements(0, s32, 6);
  LI.computeTables();

  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, LI, Observer, B);

  B.setInsertPt(*EntryMBB, EntryMBB->end());

  auto Val0 = B.buildBitcast(v2s32, Copies[0]);
  auto Val1 = B.buildBitcast(v2s32, Copies[1]);

  auto And = B.buildAnd(v2s32, Val0, Val1);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.moreElementsVector(*And, 0, v6s32));

  auto CheckStr = R"(
  CHECK: [[BITCAST0:%[0-9]+]]:_(<2 x s32>) = G_BITCAST
  CHECK: [[BITCAST1:%[0-9]+]]:_(<2 x s32>) = G_BITCAST
  CHECK: [[IMP_DEF0:%[0-9]+]]:_(<2 x s32>) = G_IMPLICIT_DEF
  CHECK: [[CONCAT0:%[0-9]+]]:_(<6 x s32>) = G_CONCAT_VECTORS [[BITCAST0]]:_(<2 x s32>), [[IMP_DEF0]]:_(<2 x s32>), [[IMP_DEF0]]:_(<2 x s32>)
  CHECK: [[IMP_DEF1:%[0-9]+]]:_(<2 x s32>) = G_IMPLICIT_DEF
  CHECK: [[CONCAT1:%[0-9]+]]:_(<6 x s32>) = G_CONCAT_VECTORS [[BITCAST1]]:_(<2 x s32>), [[IMP_DEF1]]:_(<2 x s32>), [[IMP_DEF1]]:_(<2 x s32>)
  CHECK: [[AND:%[0-9]+]]:_(<6 x s32>) = G_AND [[CONCAT0]]:_, [[CONCAT1]]:_
  CHECK: (<2 x s32>) = G_EXTRACT [[AND]]:_(<6 x s32>), 0
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, FewerElementsPhi) {
  if (!TM)
    return;

  LLT s1 = LLT::scalar(1);
  LLT s32 = LLT::scalar(32);
  LLT s64 = LLT::scalar(64);
  LLT v2s32 = LLT::vector(2, 32);
  LLT v5s32 = LLT::vector(5, 32);

  LegalizerInfo LI;
  LI.getActionDefinitionsBuilder(TargetOpcode::G_PHI)
    .legalFor({v2s32})
    .clampMinNumElements(0, s32, 2);
  LI.computeTables();

  LLT PhiTy = v5s32;
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, LI, Observer, B);
  B.setMBB(*EntryMBB);

  MachineBasicBlock *MidMBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *EndMBB = MF->CreateMachineBasicBlock();
  MF->insert(MF->end(), MidMBB);
  MF->insert(MF->end(), EndMBB);

  EntryMBB->addSuccessor(MidMBB);
  EntryMBB->addSuccessor(EndMBB);
  MidMBB->addSuccessor(EndMBB);

  auto InitVal = B.buildUndef(PhiTy);
  auto InitOtherVal = B.buildConstant(s64, 999);

  auto ICmp = B.buildICmp(CmpInst::ICMP_EQ, s1, Copies[0], Copies[1]);
  B.buildBrCond(ICmp.getReg(0), *MidMBB);
  B.buildBr(*EndMBB);


  B.setMBB(*MidMBB);
  auto MidVal = B.buildUndef(PhiTy);
  auto MidOtherVal = B.buildConstant(s64, 345);
  B.buildBr(*EndMBB);

  B.setMBB(*EndMBB);
  auto Phi = B.buildInstr(TargetOpcode::G_PHI)
    .addDef(MRI->createGenericVirtualRegister(PhiTy))
    .addUse(InitVal.getReg(0))
    .addMBB(EntryMBB)
    .addUse(MidVal.getReg(0))
    .addMBB(MidMBB);

  // Insert another irrelevant phi to make sure the rebuild is inserted after
  // it.
  B.buildInstr(TargetOpcode::G_PHI)
    .addDef(MRI->createGenericVirtualRegister(s64))
    .addUse(InitOtherVal.getReg(0))
    .addMBB(EntryMBB)
    .addUse(MidOtherVal.getReg(0))
    .addMBB(MidMBB);

  // Add some use instruction after the phis.
  B.buildAnd(PhiTy, Phi.getReg(0), Phi.getReg(0));

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.fewerElementsVector(*Phi, 0, v2s32));

  auto CheckStr = R"(
  CHECK: [[INITVAL:%[0-9]+]]:_(<5 x s32>) = G_IMPLICIT_DEF
  CHECK: [[EXTRACT0:%[0-9]+]]:_(<2 x s32>) = G_EXTRACT [[INITVAL]]:_(<5 x s32>), 0
  CHECK: [[EXTRACT1:%[0-9]+]]:_(<2 x s32>) = G_EXTRACT [[INITVAL]]:_(<5 x s32>), 64
  CHECK: [[EXTRACT2:%[0-9]+]]:_(s32) = G_EXTRACT [[INITVAL]]:_(<5 x s32>), 128
  CHECK: G_BRCOND

  CHECK: [[MIDVAL:%[0-9]+]]:_(<5 x s32>) = G_IMPLICIT_DEF
  CHECK: [[EXTRACT3:%[0-9]+]]:_(<2 x s32>) = G_EXTRACT [[MIDVAL]]:_(<5 x s32>), 0
  CHECK: [[EXTRACT4:%[0-9]+]]:_(<2 x s32>) = G_EXTRACT [[MIDVAL]]:_(<5 x s32>), 64
  CHECK: [[EXTRACT5:%[0-9]+]]:_(s32) = G_EXTRACT [[MIDVAL]]:_(<5 x s32>), 128
  CHECK: G_BR

  CHECK: [[PHI0:%[0-9]+]]:_(<2 x s32>) = G_PHI [[EXTRACT0]]:_(<2 x s32>), %bb.0, [[EXTRACT3]]:_(<2 x s32>), %bb.1
  CHECK: [[PHI1:%[0-9]+]]:_(<2 x s32>) = G_PHI [[EXTRACT1]]:_(<2 x s32>), %bb.0, [[EXTRACT4]]:_(<2 x s32>), %bb.1
  CHECK: [[PHI2:%[0-9]+]]:_(s32) = G_PHI [[EXTRACT2]]:_(s32), %bb.0, [[EXTRACT5]]:_(s32), %bb.1

  CHECK: [[OTHER_PHI:%[0-9]+]]:_(s64) = G_PHI
  CHECK: [[REBUILD_VAL_IMPDEF:%[0-9]+]]:_(<5 x s32>) = G_IMPLICIT_DEF
  CHECK: [[INSERT0:%[0-9]+]]:_(<5 x s32>) = G_INSERT [[REBUILD_VAL_IMPDEF]]:_, [[PHI0]]:_(<2 x s32>), 0
  CHECK: [[INSERT1:%[0-9]+]]:_(<5 x s32>) = G_INSERT [[INSERT0]]:_, [[PHI1]]:_(<2 x s32>), 64
  CHECK: [[INSERT2:%[0-9]+]]:_(<5 x s32>) = G_INSERT [[INSERT1]]:_, [[PHI2]]:_(s32), 128
  CHECK: [[USE_OP:%[0-9]+]]:_(<5 x s32>) = G_AND [[INSERT2]]:_, [[INSERT2]]:_
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

// FNEG expansion in terms of FSUB
TEST_F(GISelMITest, LowerFNEG) {
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FSUB).legalFor({s64});
  });

  // Build Instr. Make sure FMF are preserved.
  auto FAdd =
    B.buildInstr(TargetOpcode::G_FADD, {LLT::scalar(64)}, {Copies[0], Copies[1]},
                 MachineInstr::MIFlag::FmNsz);

  // Should not propagate the flags of src instruction.
  auto FNeg0 =
    B.buildInstr(TargetOpcode::G_FNEG, {LLT::scalar(64)}, {FAdd.getReg(0)},
                 {MachineInstr::MIFlag::FmArcp});

  // Preserve the one flag.
  auto FNeg1 =
    B.buildInstr(TargetOpcode::G_FNEG, {LLT::scalar(64)}, {Copies[0]},
                 MachineInstr::MIFlag::FmNoInfs);

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  // Perform Legalization
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.lower(*FNeg0, 0, LLT::scalar(64)));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.lower(*FNeg1, 0, LLT::scalar(64)));

  auto CheckStr = R"(
  CHECK: [[FADD:%[0-9]+]]:_(s64) = nsz G_FADD %0:_, %1:_
  CHECK: [[CONST0:%[0-9]+]]:_(s64) = G_FCONSTANT double -0.000000e+00
  CHECK: [[FSUB0:%[0-9]+]]:_(s64) = arcp G_FSUB [[CONST0]]:_, [[FADD]]:_
  CHECK: [[CONST1:%[0-9]+]]:_(s64) = G_FCONSTANT double -0.000000e+00
  CHECK: [[FSUB1:%[0-9]+]]:_(s64) = ninf G_FSUB [[CONST1]]:_, %0:_
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LowerMinMax) {
  if (!TM)
    return;

  LLT s64 = LLT::scalar(64);
  LLT v2s32 = LLT::vector(2, 32);

  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder({G_SMIN, G_SMAX, G_UMIN, G_UMAX})
      .lowerFor({s64, LLT::vector(2, s32)});
  });

  auto SMin = B.buildSMin(s64, Copies[0], Copies[1]);
  auto SMax = B.buildSMax(s64, Copies[0], Copies[1]);
  auto UMin = B.buildUMin(s64, Copies[0], Copies[1]);
  auto UMax = B.buildUMax(s64, Copies[0], Copies[1]);

  auto VecVal0 = B.buildBitcast(v2s32, Copies[0]);
  auto VecVal1 = B.buildBitcast(v2s32, Copies[1]);

  auto SMinV = B.buildSMin(v2s32, VecVal0, VecVal1);
  auto SMaxV = B.buildSMax(v2s32, VecVal0, VecVal1);
  auto UMinV = B.buildUMin(v2s32, VecVal0, VecVal1);
  auto UMaxV = B.buildUMax(v2s32, VecVal0, VecVal1);

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.lower(*SMin, 0, s64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.lower(*SMax, 0, s64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.lower(*UMin, 0, s64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.lower(*UMax, 0, s64));

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.lower(*SMinV, 0, v2s32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.lower(*SMaxV, 0, v2s32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.lower(*UMinV, 0, v2s32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.lower(*UMaxV, 0, v2s32));

  auto CheckStr = R"(
  CHECK: [[CMP0:%[0-9]+]]:_(s1) = G_ICMP intpred(slt), %0:_(s64), %1:_
  CHECK: [[SMIN:%[0-9]+]]:_(s64) = G_SELECT [[CMP0]]:_(s1), %0:_, %1:_

  CHECK: [[CMP1:%[0-9]+]]:_(s1) = G_ICMP intpred(sgt), %0:_(s64), %1:_
  CHECK: [[SMAX:%[0-9]+]]:_(s64) = G_SELECT [[CMP1]]:_(s1), %0:_, %1:_

  CHECK: [[CMP2:%[0-9]+]]:_(s1) = G_ICMP intpred(ult), %0:_(s64), %1:_
  CHECK: [[UMIN:%[0-9]+]]:_(s64) = G_SELECT [[CMP2]]:_(s1), %0:_, %1:_

  CHECK: [[CMP3:%[0-9]+]]:_(s1) = G_ICMP intpred(ugt), %0:_(s64), %1:_
  CHECK: [[UMAX:%[0-9]+]]:_(s64) = G_SELECT [[CMP3]]:_(s1), %0:_, %1:_

  CHECK: [[VEC0:%[0-9]+]]:_(<2 x s32>) = G_BITCAST %0:_(s64)
  CHECK: [[VEC1:%[0-9]+]]:_(<2 x s32>) = G_BITCAST %1:_(s64)

  CHECK: [[VCMP0:%[0-9]+]]:_(<2 x s1>) = G_ICMP intpred(slt), [[VEC0]]:_(<2 x s32>), [[VEC1]]:_
  CHECK: [[SMINV:%[0-9]+]]:_(<2 x s32>) = G_SELECT [[VCMP0]]:_(<2 x s1>), [[VEC0]]:_, [[VEC1]]:_

  CHECK: [[VCMP1:%[0-9]+]]:_(<2 x s1>) = G_ICMP intpred(sgt), [[VEC0]]:_(<2 x s32>), [[VEC1]]:_
  CHECK: [[SMAXV:%[0-9]+]]:_(<2 x s32>) = G_SELECT [[VCMP1]]:_(<2 x s1>), [[VEC0]]:_, [[VEC1]]:_

  CHECK: [[VCMP2:%[0-9]+]]:_(<2 x s1>) = G_ICMP intpred(ult), [[VEC0]]:_(<2 x s32>), [[VEC1]]:_
  CHECK: [[UMINV:%[0-9]+]]:_(<2 x s32>) = G_SELECT [[VCMP2]]:_(<2 x s1>), [[VEC0]]:_, [[VEC1]]:_

  CHECK: [[VCMP3:%[0-9]+]]:_(<2 x s1>) = G_ICMP intpred(ugt), [[VEC0]]:_(<2 x s32>), [[VEC1]]:_
  CHECK: [[UMAXV:%[0-9]+]]:_(<2 x s32>) = G_SELECT [[VCMP3]]:_(<2 x s1>), [[VEC0]]:_, [[VEC1]]:_
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, WidenScalarBuildVector) {
  if (!TM)
    return;

  LLT S32 = LLT::scalar(32);
  LLT S16 = LLT::scalar(16);
  LLT V2S16 = LLT::vector(2, S16);
  LLT V2S32 = LLT::vector(2, S32);

  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder({G_SMIN, G_SMAX, G_UMIN, G_UMAX})
      .lowerFor({s64, LLT::vector(2, s32)});
  });

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  B.setInsertPt(*EntryMBB, EntryMBB->end());

  Register Constant0 = B.buildConstant(S16, 1).getReg(0);
  Register Constant1 = B.buildConstant(S16, 2).getReg(0);
  auto BV0 = B.buildBuildVector(V2S16, {Constant0, Constant1});
  auto BV1 = B.buildBuildVector(V2S16, {Constant0, Constant1});

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.widenScalar(*BV0, 0, V2S32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.widenScalar(*BV1, 1, S32));

  auto CheckStr = R"(
  CHECK: [[K0:%[0-9]+]]:_(s16) = G_CONSTANT i16 1
  CHECK-NEXT: [[K1:%[0-9]+]]:_(s16) = G_CONSTANT i16 2
  CHECK-NEXT: [[EXT_K0_0:%[0-9]+]]:_(s32) = G_ANYEXT [[K0]]
  CHECK-NEXT: [[EXT_K1_0:%[0-9]+]]:_(s32) = G_ANYEXT [[K1]]
  CHECK-NEXT: [[BV0:%[0-9]+]]:_(<2 x s32>) = G_BUILD_VECTOR [[EXT_K0_0]]:_(s32), [[EXT_K1_0]]:_(s32)
  CHECK-NEXT: [[BV0_TRUNC:%[0-9]+]]:_(<2 x s16>) = G_TRUNC [[BV0]]

  CHECK: [[EXT_K0_1:%[0-9]+]]:_(s32) = G_ANYEXT [[K0]]
  CHECK-NEXT: [[EXT_K1_1:%[0-9]+]]:_(s32) = G_ANYEXT [[K1]]

  CHECK-NEXT: [[BV1:%[0-9]+]]:_(<2 x s16>) = G_BUILD_VECTOR_TRUNC [[EXT_K0_1]]:_(s32), [[EXT_K1_1]]:_(s32)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LowerMergeValues) {
  if (!TM)
    return;

  const LLT S32 = LLT::scalar(32);
  const LLT S24 = LLT::scalar(24);
  const LLT S21 = LLT::scalar(21);
  const LLT S16 = LLT::scalar(16);
  const LLT S9 = LLT::scalar(9);
  const LLT S8 = LLT::scalar(8);
  const LLT S3 = LLT::scalar(3);

  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_UNMERGE_VALUES)
      .widenScalarIf(typeIs(1, LLT::scalar(3)), changeTo(1, LLT::scalar(9)));
  });

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  B.setInsertPt(*EntryMBB, EntryMBB->end());

  // 24 = 3 3 3   3 3 3   3 3
  //     => 9
  //
  // This can do 3 merges, but need an extra implicit_def.
  SmallVector<Register, 8> Merge0Ops;
  for (int I = 0; I != 8; ++I)
    Merge0Ops.push_back(B.buildConstant(S3, I).getReg(0));

  auto Merge0 = B.buildMerge(S24, Merge0Ops);

  // 21 = 3 3 3   3 3 3   3
  //     => 9, 2 extra implicit_def needed
  //
  SmallVector<Register, 8> Merge1Ops;
  for (int I = 0; I != 7; ++I)
    Merge1Ops.push_back(B.buildConstant(S3, I).getReg(0));

  auto Merge1 = B.buildMerge(S21, Merge1Ops);

  SmallVector<Register, 8> Merge2Ops;
  for (int I = 0; I != 2; ++I)
    Merge2Ops.push_back(B.buildConstant(S8, I).getReg(0));

  auto Merge2 = B.buildMerge(S16, Merge2Ops);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.widenScalar(*Merge0, 1, S9));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.widenScalar(*Merge1, 1, S9));

  // Request a source size greater than the original destination size.
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.widenScalar(*Merge2, 1, S32));

  auto CheckStr = R"(
  CHECK: [[K0:%[0-9]+]]:_(s3) = G_CONSTANT i3 0
  CHECK-NEXT: [[K1:%[0-9]+]]:_(s3) = G_CONSTANT i3 1
  CHECK-NEXT: [[K2:%[0-9]+]]:_(s3) = G_CONSTANT i3 2
  CHECK-NEXT: [[K3:%[0-9]+]]:_(s3) = G_CONSTANT i3 3
  CHECK-NEXT: [[K4:%[0-9]+]]:_(s3) = G_CONSTANT i3 -4
  CHECK-NEXT: [[K5:%[0-9]+]]:_(s3) = G_CONSTANT i3 -3
  CHECK-NEXT: [[K6:%[0-9]+]]:_(s3) = G_CONSTANT i3 -2
  CHECK-NEXT: [[K7:%[0-9]+]]:_(s3) = G_CONSTANT i3 -1
  CHECK-NEXT: [[IMPDEF0:%[0-9]+]]:_(s3) = G_IMPLICIT_DEF
  CHECK-NEXT: [[MERGE0:%[0-9]+]]:_(s9) = G_MERGE_VALUES [[K0]]:_(s3), [[K1]]:_(s3), [[K2]]:_(s3)
  CHECK-NEXT: [[MERGE1:%[0-9]+]]:_(s9) = G_MERGE_VALUES [[K3]]:_(s3), [[K4]]:_(s3), [[K5]]:_(s3)
  CHECK-NEXT: [[MERGE2:%[0-9]+]]:_(s9) = G_MERGE_VALUES [[K6]]:_(s3), [[K7]]:_(s3), [[IMPDEF0]]:_(s3)
  CHECK-NEXT: [[MERGE3:%[0-9]+]]:_(s27) = G_MERGE_VALUES [[MERGE0]]:_(s9), [[MERGE1]]:_(s9), [[MERGE2]]:_(s9)
  CHECK-NEXT: (s24) = G_TRUNC [[MERGE3]]:_(s27)


  CHECK: [[K8:%[0-9]+]]:_(s3) = G_CONSTANT i3 0
  CHECK-NEXT: [[K9:%[0-9]+]]:_(s3) = G_CONSTANT i3 1
  CHECK-NEXT: [[K10:%[0-9]+]]:_(s3) = G_CONSTANT i3 2
  CHECK-NEXT: [[K11:%[0-9]+]]:_(s3) = G_CONSTANT i3 3
  CHECK-NEXT: [[K12:%[0-9]+]]:_(s3) = G_CONSTANT i3 -4
  CHECK-NEXT: [[K13:%[0-9]+]]:_(s3) = G_CONSTANT i3 -3
  CHECK-NEXT: [[K14:%[0-9]+]]:_(s3) = G_CONSTANT i3 -2
  CHECK-NEXT: [[IMPDEF1:%[0-9]+]]:_(s3) = G_IMPLICIT_DEF
  CHECK-NEXT: [[MERGE4:%[0-9]+]]:_(s9) = G_MERGE_VALUES [[K8]]:_(s3), [[K9]]:_(s3), [[K10]]:_(s3)
  CHECK-NEXT: [[MERGE5:%[0-9]+]]:_(s9) = G_MERGE_VALUES [[K11]]:_(s3), [[K12]]:_(s3), [[K13]]:_(s3)
  CHECK-NEXT: [[MERGE6:%[0-9]+]]:_(s9) = G_MERGE_VALUES [[K14]]:_(s3), [[IMPDEF1]]:_(s3), [[IMPDEF1]]:_(s3)
  CHECK-NEXT: [[MERGE7:%[0-9]+]]:_(s27) = G_MERGE_VALUES [[MERGE4]]:_(s9), [[MERGE5]]:_(s9), [[MERGE6]]:_(s9)
  CHECK-NEXT: (s21) = G_TRUNC [[MERGE7]]:_(s27)


  CHECK: [[K15:%[0-9]+]]:_(s8) = G_CONSTANT i8 0
  CHECK-NEXT: [[K16:%[0-9]+]]:_(s8) = G_CONSTANT i8 1
  CHECK-NEXT: [[ZEXT_K15:[0-9]+]]:_(s32) = G_ZEXT [[K15]]:_(s8)
  CHECK-NEXT: [[ZEXT_K16:[0-9]+]]:_(s32) = G_ZEXT [[K16]]:_(s8)
  [[K16:%[0-9]+]]:_(s32) = G_CONSTANT i32 8
  [[SHL:%[0-9]+]]:_(s32) = G_SHL [[ZEXT_K16]]:_, [[K16]]:_(s32)
  [[OR:%[0-9]+]]:_(s32) = G_OR [[ZEXT_K16]]:_, [[SHL]]:_
  (s16) = G_TRUNC [[OR]]:_(s32)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, WidenScalarMergeValuesPointer) {
  if (!TM)
    return;

  DefineLegalizerInfo(A, {});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  B.setInsertPt(*EntryMBB, EntryMBB->end());

  const LLT S32 = LLT::scalar(32);
  const LLT S64 = LLT::scalar(64);
  const LLT P0 = LLT::pointer(0, 64);

  auto Lo = B.buildTrunc(S32, Copies[0]);
  auto Hi = B.buildTrunc(S32, Copies[1]);

  auto Merge = B.buildMerge(P0, {Lo, Hi});

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.widenScalar(*Merge, 1, S64));

  auto CheckStr = R"(
   CHECK: [[TRUNC0:%[0-9]+]]:_(s32) = G_TRUNC
   CHECK: [[TRUNC1:%[0-9]+]]:_(s32) = G_TRUNC
   CHECK: [[ZEXT_TRUNC0:%[0-9]+]]:_(s64) = G_ZEXT [[TRUNC0]]
   CHECK: [[ZEXT_TRUNC1:%[0-9]+]]:_(s64) = G_ZEXT [[TRUNC1]]
   CHECK: [[SHIFT_AMT:%[0-9]+]]:_(s64) = G_CONSTANT i64 32
   CHECK: [[SHL:%[0-9]+]]:_(s64) = G_SHL [[ZEXT_TRUNC1]]:_, [[SHIFT_AMT]]
   CHECK: [[OR:%[0-9]+]]:_(s64) = G_OR [[ZEXT_TRUNC0]]:_, [[SHL]]
   CHECK: [[INTTOPTR:%[0-9]+]]:_(p0) = G_INTTOPTR [[OR]]:_(s64)
  )";

  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, WidenSEXTINREG) {
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_SEXT_INREG).legalForTypeWithAnyImm({s64});
  });
  // Build Instr
  auto MIB = B.buildInstr(
      TargetOpcode::G_SEXT_INREG, {LLT::scalar(32)},
      {B.buildInstr(TargetOpcode::G_TRUNC, {LLT::scalar(32)}, {Copies[0]}),
       uint64_t(8)});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  // Perform Legalization
  ASSERT_TRUE(Helper.widenScalar(*MIB, 0, LLT::scalar(64)) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[T0:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[T1:%[0-9]+]]:_(s64) = G_ANYEXT [[T0]]:_(s32)
  CHECK: [[T2:%[0-9]+]]:_(s64) = G_SEXT_INREG [[T1]]:_, 8
  CHECK: [[T3:%[0-9]+]]:_(s32) = G_TRUNC [[T2]]:_(s64)
  )";

  // Check
  ASSERT_TRUE(CheckMachineFunction(*MF, CheckStr));
}

TEST_F(GISelMITest, NarrowSEXTINREG) {
  if (!TM)
    return;

  // Declare your legalization info, these aren't actually relevant to the test.
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_SEXT_INREG).legalForTypeWithAnyImm({s64});
  });
  // Build Instr
  auto MIB = B.buildInstr(
      TargetOpcode::G_SEXT_INREG, {LLT::scalar(16)},
      {B.buildInstr(TargetOpcode::G_TRUNC, {LLT::scalar(16)}, {Copies[0]}),
       uint64_t(8)});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  // Perform Legalization
  ASSERT_TRUE(Helper.narrowScalar(*MIB, 0, LLT::scalar(10)) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[T0:%[0-9]+]]:_(s16) = G_TRUNC
  CHECK: [[T1:%[0-9]+]]:_(s10) = G_TRUNC [[T0]]:_(s16)
  CHECK: [[T2:%[0-9]+]]:_(s10) = G_SEXT_INREG [[T1]]:_, 8
  CHECK: [[T3:%[0-9]+]]:_(s16) = G_SEXT [[T2]]:_(s10)
  )";

  // Check
  ASSERT_TRUE(CheckMachineFunction(*MF, CheckStr));
}

TEST_F(GISelMITest, NarrowSEXTINREG2) {
  if (!TM)
    return;

  // Declare your legalization info, these aren't actually relevant to the test.
  DefineLegalizerInfo(
      A, { getActionDefinitionsBuilder(G_SEXT_INREG).legalForTypeWithAnyImm({s64}); });
  // Build Instr
  auto MIB = B.buildInstr(
      TargetOpcode::G_SEXT_INREG, {LLT::scalar(32)},
      {B.buildInstr(TargetOpcode::G_TRUNC, {LLT::scalar(32)}, {Copies[0]}),
       uint64_t(9)});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  // Perform Legalization
  ASSERT_TRUE(Helper.narrowScalar(*MIB, 0, LLT::scalar(8)) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[T0:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[T1:%[0-9]+]]:_(s8), [[T2:%[0-9]+]]:_(s8), [[T3:%[0-9]+]]:_(s8), [[T4:%[0-9]+]]:_(s8) = G_UNMERGE_VALUES [[T0]]:_(s32)
  CHECK: [[CST2:%[0-9]+]]:_(s8) = G_CONSTANT i8 7
  CHECK: [[T5:%[0-9]+]]:_(s8) = G_SEXT_INREG [[T2]]:_, 1
  CHECK: [[T6:%[0-9]+]]:_(s8) = G_ASHR [[T5]]:_, [[CST2]]:_
  CHECK: [[T7:%[0-9]+]]:_(s32) = G_MERGE_VALUES [[T1]]:_(s8), [[T5]]:_(s8), [[T6]]:_(s8), [[T6]]:_(s8)
  )";

  // Check
  ASSERT_TRUE(CheckMachineFunction(*MF, CheckStr));
}

TEST_F(GISelMITest, LowerSEXTINREG) {
  if (!TM)
    return;

  // Declare your legalization info, these aren't actually relevant to the test.
  DefineLegalizerInfo(
      A, { getActionDefinitionsBuilder(G_SEXT_INREG).legalForTypeWithAnyImm({s64}); });
  // Build Instr
  auto MIB = B.buildInstr(
      TargetOpcode::G_SEXT_INREG, {LLT::scalar(32)},
      {B.buildInstr(TargetOpcode::G_TRUNC, {LLT::scalar(32)}, {Copies[0]}),
       uint64_t(8)});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  // Perform Legalization
  ASSERT_TRUE(Helper.lower(*MIB, 0, LLT()) ==
              LegalizerHelper::LegalizeResult::Legalized);

  auto CheckStr = R"(
  CHECK: [[T1:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[CST:%[0-9]+]]:_(s32) = G_CONSTANT i32 24
  CHECK: [[T2:%[0-9]+]]:_(s32) = G_SHL [[T1]]:_, [[CST]]:_
  CHECK: [[T3:%[0-9]+]]:_(s32) = G_ASHR [[T2]]:_, [[CST]]:_
  )";

  // Check
  ASSERT_TRUE(CheckMachineFunction(*MF, CheckStr));
}

TEST_F(GISelMITest, LibcallFPExt) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FPEXT).libcallFor({{s32, s16}, {s128, s64}});
  });

  LLT S16{LLT::scalar(16)};
  LLT S32{LLT::scalar(32)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S16, Copies[0]);
  auto MIBFPExt1 =
      B.buildInstr(TargetOpcode::G_FPEXT, {S32}, {MIBTrunc});

  auto MIBFPExt2 =
      B.buildInstr(TargetOpcode::G_FPEXT, {S128}, {Copies[1]});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
              Helper.libcall(*MIBFPExt1));

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
              Helper.libcall(*MIBFPExt2));
  auto CheckStr = R"(
  CHECK: [[TRUNC:%[0-9]+]]:_(s16) = G_TRUNC
  CHECK: $h0 = COPY [[TRUNC]]
  CHECK: BL &__gnu_h2f_ieee
  CHECK: $d0 = COPY
  CHECK: BL &__extenddftf2
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFPTrunc) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FPTRUNC).libcallFor({{s16, s32}, {s64, s128}});
  });

  LLT S16{LLT::scalar(16)};
  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBFPTrunc1 =
      B.buildInstr(TargetOpcode::G_FPTRUNC, {S16}, {MIBTrunc});

  auto MIBMerge = B.buildMerge(S128, {Copies[1], Copies[2]});

  auto MIBFPTrunc2 =
      B.buildInstr(TargetOpcode::G_FPTRUNC, {S64}, {MIBMerge});
  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBFPTrunc1));

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBFPTrunc2));
  auto CheckStr = R"(
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: BL &__gnu_f2h_ieee
  CHECK: $q0 = COPY
  CHECK: BL &__trunctfdf2
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallSimple) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FADD).libcallFor({s16});
  });

  LLT S16{LLT::scalar(16)};
  auto MIBTrunc = B.buildTrunc(S16, Copies[0]);
  auto MIBFADD =
      B.buildInstr(TargetOpcode::G_FADD, {S16}, {MIBTrunc, MIBTrunc});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);
  // Make sure we do not crash anymore
  EXPECT_EQ(LegalizerHelper::LegalizeResult::UnableToLegalize,
            Helper.libcall(*MIBFADD));
}

TEST_F(GISelMITest, LibcallSRem) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_SREM).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBSRem32 =
      B.buildInstr(TargetOpcode::G_SREM, {S32}, {MIBTrunc, MIBTrunc});
  auto MIBSRem64 =
      B.buildInstr(TargetOpcode::G_SREM, {S64}, {Copies[0], Copies[0]});
  auto MIBSRem128 =
      B.buildInstr(TargetOpcode::G_SREM, {S128}, {MIBExt, MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBSRem32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBSRem64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBSRem128));

  auto CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $w0 = COPY [[TRUNC]]
  CHECK: $w1 = COPY [[TRUNC]]
  CHECK: BL &__modsi3
  CHECK: $x0 = COPY [[COPY]]
  CHECK: $x1 = COPY [[COPY]]
  CHECK: BL &__moddi3
  CHECK: [[UV:%[0-9]+]]:_(s64), [[UV1:%[0-9]+]]:_(s64) = G_UNMERGE_VALUES [[ANYEXT]]
  CHECK: [[UV2:%[0-9]+]]:_(s64), [[UV3:%[0-9]+]]:_(s64) = G_UNMERGE_VALUES [[ANYEXT]]
  CHECK: $x0 = COPY [[UV]]
  CHECK: $x1 = COPY [[UV1]]
  CHECK: $x2 = COPY [[UV2]]
  CHECK: $x3 = COPY [[UV3]]
  CHECK: BL &__modti3
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallURem) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_UREM).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBURem32 =
      B.buildInstr(TargetOpcode::G_UREM, {S32}, {MIBTrunc, MIBTrunc});
  auto MIBURem64 =
      B.buildInstr(TargetOpcode::G_UREM, {S64}, {Copies[0], Copies[0]});
  auto MIBURem128 =
      B.buildInstr(TargetOpcode::G_UREM, {S128}, {MIBExt, MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBURem32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBURem64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBURem128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $w0 = COPY [[TRUNC]]
  CHECK: $w1 = COPY [[TRUNC]]
  CHECK: BL &__umodsi3
  CHECK: $x0 = COPY [[COPY]]
  CHECK: $x1 = COPY [[COPY]]
  CHECK: BL &__umoddi3
  CHECK: [[UV:%[0-9]+]]:_(s64), [[UV1:%[0-9]+]]:_(s64) = G_UNMERGE_VALUES [[ANYEXT]]
  CHECK: [[UV2:%[0-9]+]]:_(s64), [[UV3:%[0-9]+]]:_(s64) = G_UNMERGE_VALUES [[ANYEXT]]
  CHECK: $x0 = COPY [[UV]]
  CHECK: $x1 = COPY [[UV1]]
  CHECK: $x2 = COPY [[UV2]]
  CHECK: $x3 = COPY [[UV3]]
  CHECK: BL &__umodti3
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallCtlzZeroUndef) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_CTLZ_ZERO_UNDEF)
        .libcallFor({{s32, s32}, {s64, s64}, {s128, s128}});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBCtlz32 =
      B.buildInstr(TargetOpcode::G_CTLZ_ZERO_UNDEF, {S32}, {MIBTrunc});
  auto MIBCtlz64 =
      B.buildInstr(TargetOpcode::G_CTLZ_ZERO_UNDEF, {S64}, {Copies[0]});
  auto MIBCtlz128 =
      B.buildInstr(TargetOpcode::G_CTLZ_ZERO_UNDEF, {S128}, {MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBCtlz32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBCtlz64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBCtlz128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $w0 = COPY [[TRUNC]]
  CHECK: BL &__clzsi2
  CHECK: $x0 = COPY [[COPY]]
  CHECK: BL &__clzdi2
  CHECK: [[UV:%[0-9]+]]:_(s64), [[UV1:%[0-9]+]]:_(s64) = G_UNMERGE_VALUES [[ANYEXT]]
  CHECK: $x0 = COPY [[UV]]
  CHECK: $x1 = COPY [[UV1]]
  CHECK: BL &__clzti2
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFAdd) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FADD).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBAdd32 =
      B.buildInstr(TargetOpcode::G_FADD, {S32}, {MIBTrunc, MIBTrunc});
  auto MIBAdd64 =
      B.buildInstr(TargetOpcode::G_FADD, {S64}, {Copies[0], Copies[0]});
  auto MIBAdd128 = B.buildInstr(TargetOpcode::G_FADD, {S128}, {MIBExt, MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBAdd32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBAdd64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBAdd128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: $s1 = COPY [[TRUNC]]
  CHECK: BL &__addsf3
  CHECK: $d0 = COPY [[COPY]]
  CHECK: $d1 = COPY [[COPY]]
  CHECK: BL &__adddf3
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: $q1 = COPY [[ANYEXT]]
  CHECK: BL &__addtf3
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFSub) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FSUB).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBSub32 =
      B.buildInstr(TargetOpcode::G_FSUB, {S32}, {MIBTrunc, MIBTrunc});
  auto MIBSub64 =
      B.buildInstr(TargetOpcode::G_FSUB, {S64}, {Copies[0], Copies[0]});
  auto MIBSub128 = B.buildInstr(TargetOpcode::G_FSUB, {S128}, {MIBExt, MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBSub32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBSub64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBSub128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: $s1 = COPY [[TRUNC]]
  CHECK: BL &__subsf3
  CHECK: $d0 = COPY [[COPY]]
  CHECK: $d1 = COPY [[COPY]]
  CHECK: BL &__subdf3
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: $q1 = COPY [[ANYEXT]]
  CHECK: BL &__subtf3
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFMul) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FMUL).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBMul32 =
      B.buildInstr(TargetOpcode::G_FMUL, {S32}, {MIBTrunc, MIBTrunc});
  auto MIBMul64 =
      B.buildInstr(TargetOpcode::G_FMUL, {S64}, {Copies[0], Copies[0]});
  auto MIBMul128 = B.buildInstr(TargetOpcode::G_FMUL, {S128}, {MIBExt, MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBMul32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBMul64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBMul128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: $s1 = COPY [[TRUNC]]
  CHECK: BL &__mulsf3
  CHECK: $d0 = COPY [[COPY]]
  CHECK: $d1 = COPY [[COPY]]
  CHECK: BL &__muldf3
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: $q1 = COPY [[ANYEXT]]
  CHECK: BL &__multf3
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFDiv) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FDIV).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBDiv32 =
      B.buildInstr(TargetOpcode::G_FDIV, {S32}, {MIBTrunc, MIBTrunc});
  auto MIBDiv64 =
      B.buildInstr(TargetOpcode::G_FDIV, {S64}, {Copies[0], Copies[0]});
  auto MIBDiv128 = B.buildInstr(TargetOpcode::G_FDIV, {S128}, {MIBExt, MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBDiv32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBDiv64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBDiv128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: $s1 = COPY [[TRUNC]]
  CHECK: BL &__divsf3
  CHECK: $d0 = COPY [[COPY]]
  CHECK: $d1 = COPY [[COPY]]
  CHECK: BL &__divdf3
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: $q1 = COPY [[ANYEXT]]
  CHECK: BL &__divtf3
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFExp) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FEXP).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBExp32 = B.buildInstr(TargetOpcode::G_FEXP, {S32}, {MIBTrunc});
  auto MIBExp64 = B.buildInstr(TargetOpcode::G_FEXP, {S64}, {Copies[0]});
  auto MIBExp128 = B.buildInstr(TargetOpcode::G_FEXP, {S128}, {MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBExp32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBExp64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBExp128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: BL &expf
  CHECK: $d0 = COPY [[COPY]]
  CHECK: BL &exp
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: BL &expl
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFExp2) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FEXP2).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBExp232 = B.buildInstr(TargetOpcode::G_FEXP2, {S32}, {MIBTrunc});
  auto MIBExp264 = B.buildInstr(TargetOpcode::G_FEXP2, {S64}, {Copies[0]});
  auto MIBExp2128 = B.buildInstr(TargetOpcode::G_FEXP2, {S128}, {MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBExp232));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBExp264));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBExp2128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: BL &exp2f
  CHECK: $d0 = COPY [[COPY]]
  CHECK: BL &exp2
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: BL &exp2l
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFRem) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FREM).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBFRem32 = B.buildInstr(TargetOpcode::G_FREM, {S32}, {MIBTrunc});
  auto MIBFRem64 = B.buildInstr(TargetOpcode::G_FREM, {S64}, {Copies[0]});
  auto MIBFRem128 = B.buildInstr(TargetOpcode::G_FREM, {S128}, {MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBFRem32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBFRem64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBFRem128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: BL &fmodf
  CHECK: $d0 = COPY [[COPY]]
  CHECK: BL &fmod
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: BL &fmodl
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFPow) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FPOW).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBPow32 = B.buildInstr(TargetOpcode::G_FPOW, {S32}, {MIBTrunc});
  auto MIBPow64 = B.buildInstr(TargetOpcode::G_FPOW, {S64}, {Copies[0]});
  auto MIBPow128 = B.buildInstr(TargetOpcode::G_FPOW, {S128}, {MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBPow32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBPow64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBPow128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: BL &powf
  CHECK: $d0 = COPY [[COPY]]
  CHECK: BL &pow
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: BL &powl
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFMa) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FMA).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBMa32 = B.buildInstr(TargetOpcode::G_FMA, {S32}, {MIBTrunc, MIBTrunc});
  auto MIBMa64 =
      B.buildInstr(TargetOpcode::G_FMA, {S64}, {Copies[0], Copies[0]});
  auto MIBMa128 = B.buildInstr(TargetOpcode::G_FMA, {S128}, {MIBExt, MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBMa32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBMa64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBMa128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: BL &fmaf
  CHECK: $d0 = COPY [[COPY]]
  CHECK: BL &fma
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: BL &fmal
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFCeil) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FCEIL).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBCeil32 = B.buildInstr(TargetOpcode::G_FCEIL, {S32}, {MIBTrunc});
  auto MIBCeil64 = B.buildInstr(TargetOpcode::G_FCEIL, {S64}, {Copies[0]});
  auto MIBCeil128 = B.buildInstr(TargetOpcode::G_FCEIL, {S128}, {MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBCeil32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBCeil64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBCeil128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: BL &ceilf
  CHECK: $d0 = COPY [[COPY]]
  CHECK: BL &ceil
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: BL &ceill
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFFloor) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FFLOOR).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBFloor32 = B.buildInstr(TargetOpcode::G_FFLOOR, {S32}, {MIBTrunc});
  auto MIBFloor64 = B.buildInstr(TargetOpcode::G_FFLOOR, {S64}, {Copies[0]});
  auto MIBFloor128 = B.buildInstr(TargetOpcode::G_FFLOOR, {S128}, {MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBFloor32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBFloor64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBFloor128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: BL &floorf
  CHECK: $d0 = COPY [[COPY]]
  CHECK: BL &floor
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: BL &floorl
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFMinNum) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FMINNUM).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBMin32 = B.buildFMinNum(S32, MIBTrunc, MIBTrunc);
  auto MIBMin64 = B.buildFMinNum(S64, Copies[0], Copies[0]);
  auto MIBMin128 = B.buildFMinNum(S128, MIBExt, MIBExt);

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBMin32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBMin64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBMin128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: $s1 = COPY [[TRUNC]]
  CHECK: BL &fminf
  CHECK: $d0 = COPY [[COPY]]
  CHECK: $d1 = COPY [[COPY]]
  CHECK: BL &fmin
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: $q1 = COPY [[ANYEXT]]
  CHECK: BL &fminl
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFMaxNum) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FMAXNUM).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBMax32 = B.buildFMaxNum(S32, MIBTrunc, MIBTrunc);
  auto MIBMax64 = B.buildFMaxNum(S64, Copies[0], Copies[0]);
  auto MIBMax128 = B.buildFMaxNum(S128, MIBExt, MIBExt);

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBMax32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBMax64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBMax128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: $s1 = COPY [[TRUNC]]
  CHECK: BL &fmaxf
  CHECK: $d0 = COPY [[COPY]]
  CHECK: $d1 = COPY [[COPY]]
  CHECK: BL &fmax
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: $q1 = COPY [[ANYEXT]]
  CHECK: BL &fmaxl
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFSqrt) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FSQRT).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBSqrt32 = B.buildInstr(TargetOpcode::G_FSQRT, {S32}, {MIBTrunc});
  auto MIBSqrt64 = B.buildInstr(TargetOpcode::G_FSQRT, {S64}, {Copies[0]});
  auto MIBSqrt128 = B.buildInstr(TargetOpcode::G_FSQRT, {S128}, {MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBSqrt32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBSqrt64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBSqrt128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: BL &sqrtf
  CHECK: $d0 = COPY [[COPY]]
  CHECK: BL &sqrt
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: BL &sqrtl
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFRint) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FRINT).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBRint32 = B.buildInstr(TargetOpcode::G_FRINT, {S32}, {MIBTrunc});
  auto MIBRint64 = B.buildInstr(TargetOpcode::G_FRINT, {S64}, {Copies[0]});
  auto MIBRint128 = B.buildInstr(TargetOpcode::G_FRINT, {S128}, {MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBRint32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBRint64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBRint128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: BL &rintf
  CHECK: $d0 = COPY [[COPY]]
  CHECK: BL &rint
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: BL &rintl
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, LibcallFNearbyInt) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_FNEARBYINT).libcallFor({s32, s64, s128});
  });

  LLT S32{LLT::scalar(32)};
  LLT S64{LLT::scalar(64)};
  LLT S128{LLT::scalar(128)};
  auto MIBTrunc = B.buildTrunc(S32, Copies[0]);
  auto MIBExt = B.buildAnyExt(S128, Copies[0]);

  auto MIBNearbyInt32 =
      B.buildInstr(TargetOpcode::G_FNEARBYINT, {S32}, {MIBTrunc});
  auto MIBNearbyInt64 =
      B.buildInstr(TargetOpcode::G_FNEARBYINT, {S64}, {Copies[0]});
  auto MIBNearbyInt128 =
      B.buildInstr(TargetOpcode::G_FNEARBYINT, {S128}, {MIBExt});

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBNearbyInt32));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBNearbyInt64));
  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.libcall(*MIBNearbyInt128));

  const auto *CheckStr = R"(
  CHECK: [[COPY:%[0-9]+]]:_(s64) = COPY
  CHECK: [[TRUNC:%[0-9]+]]:_(s32) = G_TRUNC
  CHECK: [[ANYEXT:%[0-9]+]]:_(s128) = G_ANYEXT
  CHECK: $s0 = COPY [[TRUNC]]
  CHECK: BL &nearbyintf
  CHECK: $d0 = COPY [[COPY]]
  CHECK: BL &nearbyint
  CHECK: $q0 = COPY [[ANYEXT]]
  CHECK: BL &nearbyintl
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}

TEST_F(GISelMITest, NarrowScalarExtract) {
  setUp();
  if (!TM)
    return;

  // Declare your legalization info
  DefineLegalizerInfo(A, {
    getActionDefinitionsBuilder(G_UNMERGE_VALUES).legalFor({{s32, s64}});
    getActionDefinitionsBuilder(G_EXTRACT).legalFor({{s16, s32}});
  });

  LLT S16{LLT::scalar(16)};
  LLT S32{LLT::scalar(32)};

  auto MIBExtractS32 = B.buildExtract(S32, Copies[1], 32);
  auto MIBExtractS16 = B.buildExtract(S16, Copies[1], 0);

  AInfo Info(MF->getSubtarget());
  DummyGISelObserver Observer;
  LegalizerHelper Helper(*MF, Info, Observer, B);

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.narrowScalar(*MIBExtractS32, 1, S32));

  EXPECT_EQ(LegalizerHelper::LegalizeResult::Legalized,
            Helper.narrowScalar(*MIBExtractS16, 1, S32));

  const auto *CheckStr = R"(
  CHECK: [[UV:%[0-9]+]]:_(s32), [[UV1:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES
  CHECK: [[COPY:%[0-9]+]]:_(s32) = COPY [[UV1]]
  CHECK: [[UV3:%[0-9]+]]:_(s32), [[UV4:%[0-9]+]]:_(s32) = G_UNMERGE_VALUES
  CHECK: [[EXTR:%[0-9]+]]:_(s16) = G_EXTRACT [[UV3]]:_(s32), 0
  CHECK: [[COPY:%[0-9]+]]:_(s16) = COPY [[EXTR]]
  )";

  // Check
  EXPECT_TRUE(CheckMachineFunction(*MF, CheckStr)) << *MF;
}
} // namespace
