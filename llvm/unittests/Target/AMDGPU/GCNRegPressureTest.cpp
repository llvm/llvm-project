//===- GCNRegPressureTest.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GCNRegPressure.h"
#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Passes/PassBuilder.h"
#include "gtest/gtest.h"

using namespace llvm;

class GCNRegPressureTest : public llvm::CodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgpu9.08--", "", ""); }
};

TEST_F(GCNRegPressureTest, DownwardTrackerEndOnDbgVal) {
  StringRef MIR = R"(
name:            DownwardTrackerEndOnDbgVal
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = IMPLICIT_DEF
    %1:vgpr_32 = IMPLICIT_DEF
  
  bb.1:
    DBG_VALUE %0
    DBG_VALUE %1
    %2:vgpr_32 = IMPLICIT_DEF
  
  bb.3:
    S_NOP 0, implicit %0, implicit %1, implicit %2
    S_ENDPGM 0
...
)";
  EXPECT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("DownwardTrackerEndOnDbgVal");
  const LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

  // MBB1 live-in pressure is equivalent to MBB0 live-out pressure.
  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  MachineBasicBlock &MBB1 = *MF.getBlockNumbered(1);
  GCNRPTracker::LiveRegSet MBB1LiveIns =
      getLiveRegs(LIS.getInstructionIndex(*MBB0.rbegin()).getDeadSlot(), LIS,
                  MF.getRegInfo());

  // Track pressure across MBB1.
  {
    GCNDownwardRPTracker RPTracker(LIS), RPTrackerNoLiveIns(LIS);

    // There is a non-debug instruction in bb.1 (%2's def), so advance should
    // return true.
    EXPECT_TRUE(RPTracker.advance(MBB1.begin(), MBB1.end(), &MBB1LiveIns));
    EXPECT_TRUE(RPTrackerNoLiveIns.advance(MBB1.begin(), MBB1.end(), nullptr));

    // When advance returns true, maximum pressure should be the pressured
    // induced by the block's live-ins plus %2's def i.e., 3 VGPRs.
    EXPECT_EQ(RPTracker.moveMaxPressure().getVGPRNum(false), 3U);
    EXPECT_EQ(RPTrackerNoLiveIns.moveMaxPressure().getVGPRNum(false), 3U);
  }

  // Track pressure just across the first debug value of bb.1.
  {
    MachineBasicBlock::iterator Dbg1 = std::next(MBB1.begin());
    GCNDownwardRPTracker RPTracker(LIS), RPTrackerNoLiveIns(LIS);

    // The following unpacks a call to
    // advance(*MBB1.begin(), Dbg1, [MBB1LiveIns|nullptr])
    // which would return false in this case.
    //
    // There aren't any non-debug instruction between the beginning of bb1 and
    // Dbg1 (exclusive), the reset is therefore unsuccessful. The advance caller
    // returns early on a failure to reset. Calling advance after this does
    // nothing and produces false because the internal iterator already points
    // to the second debug instruction.
    EXPECT_FALSE(RPTracker.reset(*MBB1.begin(), Dbg1, &MBB1LiveIns));
    EXPECT_FALSE(RPTrackerNoLiveIns.reset(*MBB1.begin(), Dbg1, nullptr));
    EXPECT_FALSE(RPTracker.advance(Dbg1));
    EXPECT_FALSE(RPTrackerNoLiveIns.advance(Dbg1));

    // Register pressure should be the one at the block's live-ins.
    EXPECT_EQ(RPTracker.moveMaxPressure().getVGPRNum(false), 2U);
    EXPECT_EQ(RPTrackerNoLiveIns.moveMaxPressure().getVGPRNum(false), 2U);
  }
}

TEST_F(GCNRegPressureTest, DownwardTrackerAllDbgVal) {
  StringRef MIR = R"(
name:            DownwardTrackerAllDbgVal
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = IMPLICIT_DEF

  bb.1:
    DBG_VALUE %0
  
  bb.2:
    S_NOP 0, implicit %0
    S_ENDPGM 0
...
)";
  EXPECT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("DownwardTrackerAllDbgVal");
  const LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

  // MBB1 live-in pressure is equivalent to MBB0 live-out pressure.
  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  GCNRPTracker::LiveRegSet MBB1LiveIns =
      getLiveRegs(LIS.getInstructionIndex(*MBB0.rbegin()).getDeadSlot(), LIS,
                  MF.getRegInfo());

  MachineBasicBlock &MBB1 = *MF.getBlockNumbered(1);
  GCNDownwardRPTracker RPTracker(LIS), RPTrackerNoLiveIns(LIS);

  // The following unpacks a call to
  // advance(MBB1.begin(), MBB1.end(), [MBB1LiveIns|nullptr])
  // which would return false in this case.
  //
  // There aren't any non-debug instruction in bb.2, the reset is therefore
  // unsuccessful. The advance caller returns early on a failure to reset.
  // Calling advance after this does nothing and produces false because the
  // internal iterator is already at the block's end.
  EXPECT_FALSE(RPTracker.reset(*MBB1.begin(), MBB1.end(), &MBB1LiveIns));
  EXPECT_FALSE(RPTrackerNoLiveIns.reset(*MBB1.begin(), MBB1.end(), nullptr));
  EXPECT_FALSE(RPTracker.advance(MBB1.end()));
  EXPECT_FALSE(RPTrackerNoLiveIns.advance(MBB1.end()));

  // Register pressure should be the one at the block's live-ins.
  EXPECT_EQ(RPTracker.moveMaxPressure().getVGPRNum(false), 1U);
  EXPECT_EQ(RPTrackerNoLiveIns.moveMaxPressure().getVGPRNum(false), 1U);
}

// Tests the correct handling of multiple uses of the same virtual register
// in bumpDownwardPressure (speculative estimate of register pressure).
TEST_F(GCNRegPressureTest, BumpDownwardPressureLastUseAfterCommit) {
  StringRef MIR = R"(
name:            BumpDownwardPressureLastUseAfterCommit
tracksRegLiveness: true
body:             |
  bb.0:
    %0:vgpr_32 = IMPLICIT_DEF
    %1:vreg_256_align2 = IMPLICIT_DEF
    S_NOP 0, implicit %1
    S_NOP 0, implicit %1
    S_NOP 0, implicit %0
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("BumpDownwardPressureLastUseAfterCommit");
  const LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo *TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();

  MachineBasicBlock &MBB = *MF.getBlockNumbered(0);

  SmallVector<MachineInstr *, 8> Instrs;
  for (MachineInstr &MI : MBB)
    Instrs.push_back(&MI);
  // 0: def %0, 1: def %1, 2: U1 (use %1), 3: U2 (last use %1),
  // 4: use %0, 5: S_ENDPGM
  MachineInstr *DefV0 = Instrs[0];
  MachineInstr *DefV1 = Instrs[1];
  MachineInstr *U1 = Instrs[2];
  MachineInstr *U2 = Instrs[3];

  GCNDownwardRPTracker RPTracker(LIS);
  GCNRPTracker::LiveRegSet Empty;
  RPTracker.reset(MRI, Empty);

  // Commit the defs and the first use of %1 via the externally-managed
  // iterator (same as while scheduling).
  RPTracker.advance(DefV0, /*UseInternalIterator=*/false);
  RPTracker.advance(DefV1, /*UseInternalIterator=*/false);
  RPTracker.advance(U1, /*UseInternalIterator=*/false);

  // After committing U1, both %0 (1 VGPR) and %1 (vreg_256 = 8 VGPRs) are live.
  EXPECT_EQ(RPTracker.getPressure().getArchVGPRNum(), 9U);

  // Speculate the last use of %1. %1 must die here, dropping its 8 VGPRs and
  // leaving only %0 live.
  GCNRegPressure P = RPTracker.bumpDownwardPressure(U2, TRI);
  EXPECT_EQ(P.getArchVGPRNum(), 1U);
}
