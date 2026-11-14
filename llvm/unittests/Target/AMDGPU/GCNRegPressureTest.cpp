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

class GCNRegPressureTest : public AMDGPUCodeGenTestBase {
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
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

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
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

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
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
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

  GCNManualDownwardRPTracker RPTracker(LIS);
  GCNRPTracker::LiveRegSet Empty;
  RPTracker.reset(MRI, Empty);

  // Commit the defs and the first use of %1 via the externally-managed
  // iterator (same as while scheduling).
  RPTracker.advance(*DefV0);
  RPTracker.advance(*DefV1);
  RPTracker.advance(*U1);

  // After committing U1, both %0 (1 VGPR) and %1 (vreg_256 = 8 VGPRs) are live.
  EXPECT_EQ(RPTracker.getPressure().getArchVGPRNum(), 9U);

  // Speculate the last use of %1. %1 must die here, dropping its 8 VGPRs and
  // leaving only %0 live.
  GCNRegPressure P = RPTracker.bumpDownwardPressure(U2, TRI);
  EXPECT_EQ(P.getArchVGPRNum(), 1U);
}

// Tests bumpDownwardPressure for an instruction that uses and redefines
// the same register.
TEST_F(GCNRegPressureTest, BumpDownwardPressureUseAndRedef) {
  StringRef MIR = R"(
name:            BumpDownwardPressureUseAndRedef
tracksRegLiveness: true
body:             |
  bb.0:
    %0:sgpr_32 = IMPLICIT_DEF
    %0:sgpr_32 = S_OR_B32 %0:sgpr_32, 1, implicit-def dead $scc
    S_NOP 0, implicit %0
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("BumpDownwardPressureUseAndRedef");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const SIRegisterInfo *TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();

  MachineBasicBlock &MBB = *MF.getBlockNumbered(0);

  SmallVector<MachineInstr *, 8> Instrs;
  for (MachineInstr &MI : MBB)
    Instrs.push_back(&MI);
  // 0: def %0 (sgpr_32 = 1 SGPR)
  // 1: use+redef of %0
  // 2: use %0
  // 3: S_ENDPGM
  MachineInstr *DefS0 = Instrs[0];
  MachineInstr *UseRedef = Instrs[1];

  GCNManualDownwardRPTracker RPTracker(LIS);
  GCNRPTracker::LiveRegSet Empty;
  RPTracker.reset(MRI, Empty);

  // Commit the def; %0 occupies 1 SGPR and stays live.
  RPTracker.advance(*DefS0);
  EXPECT_EQ(RPTracker.getPressure().getSGPRNum(), 1U);

  // Speculate the instruction. It uses and redefines %0, which stays live, so
  // pressure must be unchanged.
  GCNRegPressure P = RPTracker.bumpDownwardPressure(UseRedef, TRI);
  EXPECT_EQ(P.getSGPRNum(), 1U);
}

// Tests the banked pressure accounting for the av_* classes, whose values the
// allocator may place in either bank.
class GCNRegPressureBankingTest : public AMDGPUCodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgpu9.42-amd-amdhsa", "", ""); }

protected:
  using Banked = std::pair<unsigned, unsigned>;

  /// \returns the maximum pressure over the single basic block of the function
  /// named \p FnName in \p MIR.
  GCNRegPressure maxPressureOf(StringRef MIR, StringRef FnName) {
    EXPECT_TRUE(parseMIR(MIR));
    MachineFunction &MF = getMF(FnName);
    LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
    MachineBasicBlock &MBB = *MF.getBlockNumbered(0);
    GCNDownwardRPTracker RPTracker(LIS);
    EXPECT_TRUE(RPTracker.advance(MBB.begin(), MBB.end(), nullptr));
    return RPTracker.moveMaxPressure();
  }
};

TEST_F(GCNRegPressureBankingTest, BankedVGPRNumDistributesAVGPRs) {
  StringRef MIR = R"(
name:            banked_distribute
tracksRegLiveness: true
body:             |
  bb.0:
    %0:vreg_128 = IMPLICIT_DEF
    %1:areg_64 = IMPLICIT_DEF
    %2:av_256 = IMPLICIT_DEF
    S_NOP 0, implicit %0, implicit %1, implicit %2
    S_ENDPGM 0
...
)";
  GCNRegPressure RP = maxPressureOf(MIR, "banked_distribute");

  // 4 rigid ArchVGPRs, 2 rigid AGPRs, and 8 that may go to either bank.
  ASSERT_EQ(RP.getNumRegs(GCNRegPressure::VGPR), 4U);
  ASSERT_EQ(RP.getNumRegs(GCNRegPressure::AGPR), 2U);
  ASSERT_EQ(RP.getNumRegs(GCNRegPressure::AVGPR), 8U);

  // No headroom in the ArchVGPR bank, so every flexible value goes to AGPR.
  EXPECT_EQ(RP.getBankedVGPRNum(0, 256), Banked(4, 10));
  EXPECT_EQ(RP.getBankedVGPRNum(4, 256), Banked(4, 10));

  // A limit below the rigid ArchVGPR pressure must clamp the headroom to zero
  // rather than wrap around.
  EXPECT_EQ(RP.getBankedVGPRNum(2, 256), Banked(4, 10));

  // Partial headroom: 2 of the 8 flexible values fit, the rest go to AGPR.
  EXPECT_EQ(RP.getBankedVGPRNum(6, 256), Banked(6, 8));

  // Headroom exactly absorbs the flexible values.
  EXPECT_EQ(RP.getBankedVGPRNum(12, 256), Banked(12, 2));

  // With room to spare the result matches the unbanked accounting, which
  // charges every AVGPR to the ArchVGPR bank.
  EXPECT_EQ(RP.getBankedVGPRNum(256, 256), Banked(12, 2));
  EXPECT_EQ(RP.getBankedVGPRNum(256, 256),
            Banked(RP.getArchVGPRNum(), RP.getAGPRNum()));

  // Partial AGPR headroom: 2 of the 8 flexible values fit in the ArchVGPR
  // bank, 3 in the AGPR bank, and the remaining 3 are charged back to the
  // ArchVGPR bank as excess.
  EXPECT_EQ(RP.getBankedVGPRNum(6, 5), Banked(9, 5));

  // No AGPRs allocatable: every flexible value stays in the ArchVGPR bank.
  EXPECT_EQ(RP.getBankedVGPRNum(4, 0), Banked(12, 2));
  EXPECT_EQ(RP.getBankedVGPRNum(6, 0), Banked(12, 2));

  // An AGPR limit below the rigid AGPR pressure must clamp the headroom to
  // zero rather than wrap around.
  EXPECT_EQ(RP.getBankedVGPRNum(4, 1), Banked(12, 2));
}

TEST_F(GCNRegPressureBankingTest, BankedVGPRNumWithoutAVGPRs) {
  StringRef MIR = R"(
name:            banked_no_avgprs
tracksRegLiveness: true
body:             |
  bb.0:
    %0:vreg_128 = IMPLICIT_DEF
    %1:areg_64 = IMPLICIT_DEF
    S_NOP 0, implicit %0, implicit %1
    S_ENDPGM 0
...
)";
  GCNRegPressure RP = maxPressureOf(MIR, "banked_no_avgprs");

  ASSERT_EQ(RP.getNumRegs(GCNRegPressure::AVGPR), 0U);

  // With nothing flexible to place, banking must be an exact no-op at every
  // limit.
  for (unsigned ArchLimit : {0U, 1U, 4U, 5U, 256U})
    EXPECT_EQ(RP.getBankedVGPRNum(ArchLimit, 256),
              Banked(RP.getArchVGPRNum(), RP.getAGPRNum()))
        << "ArchLimit = " << ArchLimit;
}

TEST_F(GCNRegPressureBankingTest, BankedVGPRNumMatchesReferenceSweep) {
  StringRef MIR = R"(
name:            banked_sweep
tracksRegLiveness: true
body:             |
  bb.0:
    %0:vreg_128 = IMPLICIT_DEF
    %1:areg_64 = IMPLICIT_DEF
    %2:av_256 = IMPLICIT_DEF
    S_NOP 0, implicit %0, implicit %1, implicit %2
    S_ENDPGM 0
...
)";
  GCNRegPressure RP = maxPressureOf(MIR, "banked_sweep");

  unsigned Arch = RP.getNumRegs(GCNRegPressure::VGPR);
  unsigned Acc = RP.getNumRegs(GCNRegPressure::AGPR);
  unsigned Flex = RP.getNumRegs(GCNRegPressure::AVGPR);

  // Check the closed form against placing the flexible values one at a time.
  for (unsigned ArchLimit = 0; ArchLimit <= 32; ++ArchLimit) {
    for (unsigned AccLimit = 0; AccLimit <= 32; ++AccLimit) {
      unsigned ExpectedArch = Arch, ExpectedAcc = Acc;
      for (unsigned I = 0; I != Flex; ++I) {
        if (ExpectedArch < ArchLimit)
          ++ExpectedArch;
        else if (ExpectedAcc < AccLimit)
          ++ExpectedAcc;
        else
          ++ExpectedArch;
      }
      EXPECT_EQ(RP.getBankedVGPRNum(ArchLimit, AccLimit),
                Banked(ExpectedArch, ExpectedAcc))
          << "ArchLimit = " << ArchLimit << ", AccLimit = " << AccLimit;
    }
  }
}

// Returns the maximum pressure over the only block of \p MF, from the downward
// and from the upward tracker.
static std::pair<GCNRegPressure, GCNRegPressure>
getMaxPressureOfBlock(MachineFunction &MF, LiveIntervals &LIS) {
  MachineBasicBlock &MBB = *MF.getBlockNumbered(0);

  GCNDownwardRPTracker Down(LIS);
  Down.advance(MBB.begin(), MBB.end(), nullptr);

  GCNUpwardRPTracker Up(LIS);
  Up.reset(MF.getRegInfo(),
           LIS.getSlotIndexes()->getMBBEndIdx(&MBB).getPrevSlot());
  for (const MachineInstr &MI : reverse(MBB))
    Up.recede(MI);

  return {Down.moveMaxPressure(), Up.getMaxPressure()};
}

// An AV register and a VGPR are live at different points. Only one vector
// register is live at any point.
TEST_F(GCNRegPressureTest, MaxPressureAVAndVGPRAtDifferentPoints) {
  StringRef MIR = R"(
name:            MaxPressureAVAndVGPRAtDifferentPoints
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:av_32 = IMPLICIT_DEF
    S_NOP 0, implicit %0
    %1:vgpr_32 = IMPLICIT_DEF
    S_NOP 0, implicit %1
    S_ENDPGM 0
...
)";
  EXPECT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("MaxPressureAVAndVGPRAtDifferentPoints");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

  auto [Down, Up] = getMaxPressureOfBlock(MF, LIS);
  for (const GCNRegPressure *RP : {&Down, &Up}) {
    EXPECT_EQ(RP->getArchVGPRNum(), 1U);
    EXPECT_EQ(RP->getVGPRNum(/*UnifiedVGPRFile=*/false), 1U);
  }
}

// The same with 64-bit registers, for the tuple weights.
TEST_F(GCNRegPressureTest, MaxPressureAVAndVGPRTuplesAtDifferentPoints) {
  StringRef MIR = R"(
name:            MaxPressureAVAndVGPRTuplesAtDifferentPoints
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:av_64 = IMPLICIT_DEF
    S_NOP 0, implicit %0
    %1:vreg_64 = IMPLICIT_DEF
    S_NOP 0, implicit %1
    S_ENDPGM 0
...
)";
  EXPECT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("MaxPressureAVAndVGPRTuplesAtDifferentPoints");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  unsigned TupleWeight =
      std::max(TRI->getRegClassWeight(&AMDGPU::AV_64RegClass).RegWeight,
               TRI->getRegClassWeight(&AMDGPU::VReg_64RegClass).RegWeight);

  auto [Down, Up] = getMaxPressureOfBlock(MF, LIS);
  for (const GCNRegPressure *RP : {&Down, &Up}) {
    EXPECT_EQ(RP->getArchVGPRNum(), 2U);
    EXPECT_EQ(RP->getVGPRTuplesWeight(), TupleWeight);
  }
}
