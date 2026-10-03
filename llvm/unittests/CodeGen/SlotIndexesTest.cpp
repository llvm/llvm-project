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

/// Compacts on behalf of a caller whose only index holder is \p LIS.
static unsigned compact(LiveIntervals &LIS) {
  SmallVector<SlotIndex, 0> Referenced;
  LIS.appendReferencedIndexes(Referenced);
  return LIS.getSlotIndexes()->compactIndexes(Referenced);
}

/// Leftover entries inflate live range sizes, which is what made the allocator
/// rank ranges incorrectly. Compaction must restore the real def-to-use size.
TEST_F(SlotIndexesTest, CompactIndexesShrinksSizeToRealDistance) {
  StringRef MIRString = R"(
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
  ASSERT_TRUE(parseMIR(MIRString));

  MachineFunction &MF = getMF("func");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  MachineBasicBlock &MBB = *MF.getBlockNumbered(0);

  const Register Tracked = Register::index2VirtReg(0);
  // Four instructions separate the def of %0 from its use.
  EXPECT_EQ(LIS.getInterval(Tracked).getSize(), 4u * SlotIndex::InstrDist);

  // Delete the three unrelated defs sitting between them.
  for (Register Reg : {Register::index2VirtReg(1), Register::index2VirtReg(2),
                       Register::index2VirtReg(3)})
    deleteInstr(*MF.getRegInfo().getVRegDef(Reg), LIS);
  ASSERT_EQ(std::distance(MBB.begin(), MBB.end()), 3L);

  // Deleting them does not renumber anything, so the size is still as if the
  // instructions were there.
  EXPECT_EQ(LIS.getInterval(Tracked).getSize(), 4u * SlotIndex::InstrDist);

  EXPECT_EQ(compact(LIS), 3u);

  // Only the use remains between the def and the end of the range.
  EXPECT_EQ(LIS.getInterval(Tracked).getSize(), 1u * SlotIndex::InstrDist);
  EXPECT_TRUE(MF.verify(&LIS, LIS.getSlotIndexes(), /*Banner=*/nullptr,
                        /*OS=*/&errs(), /*AbortOnError=*/false));
}

/// A referenced entry must survive: here the use of %0 is taken out of the
/// index maps while %0's live range still ends on it.
TEST_F(SlotIndexesTest, CompactIndexesKeepsReferencedEntries) {
  StringRef MIRString = R"(
---
name: func
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = IMPLICIT_DEF
    S_NOP 0, implicit %0
    S_ENDPGM 0
...
  )";
  ASSERT_TRUE(parseMIR(MIRString));

  MachineFunction &MF = getMF("func");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  MachineBasicBlock &MBB = *MF.getBlockNumbered(0);

  const Register Tracked = Register::index2VirtReg(0);
  const SlotIndex End = LIS.getInterval(Tracked).endIndex();

  // Drop the use from the maps but leave %0's range ending on its slot.
  MachineInstr &Use = *std::next(MBB.begin());
  LIS.RemoveMachineInstrFromMaps(Use);

  EXPECT_EQ(compact(LIS), 0u);
  EXPECT_EQ(LIS.getInterval(Tracked).endIndex(), End);
}

/// Block boundary entries carry no instruction by design, so compaction must
/// not mistake them for leftovers.
TEST_F(SlotIndexesTest, CompactIndexesKeepsBlockBoundaries) {
  StringRef MIRString = R"(
---
name: func
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = IMPLICIT_DEF
    %1:vgpr_32 = IMPLICIT_DEF
    S_CMP_EQ_U32 0, 0, implicit-def $scc
    S_CBRANCH_SCC1 %bb.2, implicit $scc

  bb.1:
    %2:vgpr_32 = IMPLICIT_DEF
    S_NOP 0, implicit %0

  bb.2:
    S_ENDPGM 0
...
  )";
  ASSERT_TRUE(parseMIR(MIRString));

  MachineFunction &MF = getMF("func");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  SlotIndexes &SI = *LIS.getSlotIndexes();

  // Delete one dead def in each of the first two blocks.
  deleteInstr(*MF.getRegInfo().getVRegDef(Register::index2VirtReg(1)), LIS);
  deleteInstr(*MF.getRegInfo().getVRegDef(Register::index2VirtReg(2)), LIS);

  EXPECT_EQ(compact(LIS), 2u);

  for (MachineBasicBlock &MBB : MF) {
    SlotIndex Start = LIS.getMBBStartIdx(&MBB);
    EXPECT_TRUE(Start.isValid());
    EXPECT_EQ(SI.getMBBFromIndex(Start), &MBB);
    EXPECT_LT(Start, LIS.getMBBEndIdx(&MBB));
  }

  EXPECT_TRUE(MF.verify(&LIS, &SI, /*Banner=*/nullptr, /*OS=*/&errs(),
                        /*AbortOnError=*/false));
}

/// An unreported index still has to compare sanely, since only an asserts build
/// diagnoses it. An erased entry takes the index of the surviving entry behind
/// it, so it stays inside the function and in program order, but stops
/// comparing distinct from that entry.
TEST_F(SlotIndexesTest, CompactIndexesKeepsStaleIndexesInProgramOrder) {
  StringRef MIRString = R"(
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
  ASSERT_TRUE(parseMIR(MIRString));

  MachineFunction &MF = getMF("func");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  SlotIndexes &SI = *LIS.getSlotIndexes();
  MachineBasicBlock &MBB = *MF.getBlockNumbered(0);

  // The def of %0 is the entry behind all three that are about to go away.
  const SlotIndex Behind = LIS.getInstructionIndex(*MBB.begin()).getBaseIndex();
  SmallVector<SlotIndex, 3> Stale;
  for (unsigned I : {1u, 2u, 3u}) {
    MachineInstr &MI = *MF.getRegInfo().getVRegDef(Register::index2VirtReg(I));
    Stale.push_back(LIS.getInstructionIndex(MI).getBaseIndex());
  }
  for (unsigned I : {1u, 2u, 3u})
    deleteInstr(*MF.getRegInfo().getVRegDef(Register::index2VirtReg(I)), LIS);

  ASSERT_EQ(compact(LIS), 3u);

  for (SlotIndex S : Stale) {
    // Still inside the function, rather than drifting past its end.
    EXPECT_LT(S, SI.getLastIndex());
    // Ordered with the entry behind, but no longer distinct from it.
    EXPECT_FALSE(S < Behind);
    EXPECT_FALSE(Behind < S);
    EXPECT_FALSE(S == Behind);
  }
}

/// No single caller can check that every holder reported its indexes, so an
/// unreported one is reported at the point of use rather than followed.
#if !defined(NDEBUG) && defined(GTEST_HAS_DEATH_TEST)
TEST_F(SlotIndexesTest, CompactIndexesFlagsUnreportedStaleIndex) {
  StringRef MIRString = R"(
---
name: func
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = IMPLICIT_DEF
    %1:vgpr_32 = IMPLICIT_DEF
    S_NOP 0, implicit %0
    S_ENDPGM 0
...
  )";
  ASSERT_TRUE(parseMIR(MIRString));

  MachineFunction &MF = getMF("func");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  SlotIndexes &SI = *LIS.getSlotIndexes();

  MachineInstr &Dead = *MF.getRegInfo().getVRegDef(Register::index2VirtReg(1));
  // An outside holder that forgets to report this index, the way
  // LiveDebugVariables used to.
  const SlotIndex Stale = LIS.getInstructionIndex(Dead).getBaseIndex();
  deleteInstr(Dead, LIS);

  ASSERT_EQ(compact(LIS), 1u);

  EXPECT_DEATH((void)SI.getInstructionFromIndex(Stale),
               "SlotIndex outlived the instruction it pointed at");
}
#endif
