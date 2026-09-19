//===- SlotIndexesTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SlotIndexes.h"
#include "CodeGenTestBase.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Config/Targets.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;

class SlotIndexesTest : public CodeGenTestBase {
public:
  static void SetUpTestCase() {
#if LLVM_HAS_AMDGPU_TARGET
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
#else
    GTEST_SKIP();
#endif
  }

  void SetUp() override { setUpImpl("amdgpu9.50--", "", ""); }
};

/// Deletes \p MI the way a real pass does: its index list entry survives.
static void deleteInstr(MachineInstr &MI, LiveIntervals &LIS) {
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.getReg().isVirtual() && LIS.hasInterval(MO.getReg()))
      LIS.removeInterval(MO.getReg());
  LIS.RemoveMachineInstrFromMaps(MI);
  MI.eraseFromParent();
}

static StringRef FourDefsOneUse = R"(
---
name: func
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = IMPLICIT_DEF
    %1:vgpr_32 = IMPLICIT_DEF
    %2:vgpr_32 = IMPLICIT_DEF
    %3:vgpr_32 = IMPLICIT_DEF
    S_NOP 0, implicit %0
    S_ENDPGM 0
...
)";

/// A plain distance keeps counting erased instructions; the span does not.
TEST_F(SlotIndexesTest, RealInstrSpanIgnoresDeletedInstructions) {
  ASSERT_TRUE(parseMIR(FourDefsOneUse));

  MachineFunction &MF = getMF("func");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  const SlotIndexes &SI = *LIS.getSlotIndexes();

  const Register Tracked = Register::index2VirtReg(0);
  const LiveInterval &LI = LIS.getInterval(Tracked);
  const SlotIndex Begin = LI.beginIndex(), End = LI.endIndex();

  // Four instructions from def to use, nothing deleted, so both agree.
  EXPECT_EQ(LI.getSize(), 4u * SlotIndex::InstrDist);
  EXPECT_EQ(SI.getRealInstrSpan(Begin, End), 4u * SlotIndex::InstrDist);

  // Delete the three unrelated defs sitting between them.
  for (unsigned I : {1u, 2u, 3u})
    deleteInstr(*MF.getRegInfo().getVRegDef(Register::index2VirtReg(I)), LIS);

  // Deleting renumbers nothing, so only the real span drops.
  EXPECT_EQ(LI.getSize(), 4u * SlotIndex::InstrDist);
  EXPECT_EQ(SI.getRealInstrSpan(Begin, End), 1u * SlotIndex::InstrDist);
}

/// Walked rather than cached, so it stays right as the index list changes
/// during allocation: removing and reinserting leaves the old entry dead.
TEST_F(SlotIndexesTest, RealInstrSpanFollowsTheIndexList) {
  ASSERT_TRUE(parseMIR(FourDefsOneUse));

  MachineFunction &MF = getMF("func");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  const SlotIndexes &SI = *LIS.getSlotIndexes();

  const LiveInterval &LI = LIS.getInterval(Register::index2VirtReg(0));
  const SlotIndex Begin = LI.beginIndex(), End = LI.endIndex();
  ASSERT_EQ(SI.getRealInstrSpan(Begin, End), 4u * SlotIndex::InstrDist);

  MachineInstr &MI = *MF.getRegInfo().getVRegDef(Register::index2VirtReg(1));
  LIS.RemoveMachineInstrFromMaps(MI);
  EXPECT_EQ(SI.getRealInstrSpan(Begin, End), 3u * SlotIndex::InstrDist);

  LIS.InsertMachineInstrInMaps(MI);
  EXPECT_EQ(SI.getRealInstrSpan(Begin, End), 4u * SlotIndex::InstrDist);
}

/// An empty or reversed range spans nothing.
TEST_F(SlotIndexesTest, RealInstrSpanOfEmptyRange) {
  ASSERT_TRUE(parseMIR(FourDefsOneUse));

  MachineFunction &MF = getMF("func");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  const SlotIndexes &SI = *LIS.getSlotIndexes();

  const LiveInterval &LI = LIS.getInterval(Register::index2VirtReg(0));
  const SlotIndex Begin = LI.beginIndex(), End = LI.endIndex();

  EXPECT_EQ(SI.getRealInstrSpan(Begin, Begin), 0u);
  EXPECT_EQ(SI.getRealInstrSpan(End, Begin), 0u);
}
