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
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Config/Targets.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

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

  void SetUp() override { setUpImpl("amdgcn--", "", ""); }

  /// Erases \p MI the way codegen passes do, leaving its entry behind.
  static void erase(MachineInstr &MI, SlotIndexes &SI) {
    SI.removeMachineInstrFromMaps(MI);
    MI.eraseFromParent();
  }
};

constexpr StringRef TwoBlockMIR = R"(
---
name: func
tracksRegLiveness: true
body:             |
  bb.0:
    S_NOP 0
    S_NOP 1
    S_NOP 2

  bb.1:
    S_NOP 3
    S_NOP 4
    S_ENDPGM 0
...
)";

// The first block's start is the first list entry and the last block's end is
// the only boundary that is not also a block start.
TEST_F(SlotIndexesTest, BoundariesAreNotStale) {
  ASSERT_TRUE(parseMIR(TwoBlockMIR));
  MachineFunction &MF = getMF("func");
  SlotIndexes &SI = MFAM.getResult<SlotIndexesAnalysis>(MF);

  for (MachineBasicBlock &MBB : MF) {
    SlotIndex Start = SI.getMBBStartIdx(&MBB);
    SlotIndex End = SI.getMBBEndIdx(&MBB);
    EXPECT_TRUE(SI.isBlockBoundaryIndex(Start));
    EXPECT_TRUE(SI.isBlockBoundaryIndex(End));
    EXPECT_FALSE(SI.isStaleIndex(Start));
    EXPECT_FALSE(SI.isStaleIndex(End));
    EXPECT_EQ(SI.canonicalizeIndex(Start), Start);
    EXPECT_EQ(SI.canonicalizeIndex(End), End);
  }
}

TEST_F(SlotIndexesTest, LiveIndexesAreUnchanged) {
  ASSERT_TRUE(parseMIR(TwoBlockMIR));
  MachineFunction &MF = getMF("func");
  SlotIndexes &SI = MFAM.getResult<SlotIndexesAnalysis>(MF);

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      SlotIndex Base = SI.getInstructionIndex(MI);
      EXPECT_FALSE(SI.isBlockBoundaryIndex(Base));
      EXPECT_FALSE(SI.isStaleIndex(Base));
      for (SlotIndex Idx :
           {Base, Base.getRegSlot(true), Base.getRegSlot(), Base.getDeadSlot()})
        EXPECT_EQ(SI.canonicalizeIndex(Idx), Idx);
    }
  }
}

TEST_F(SlotIndexesTest, ErasedInstrResolvesToPrecedingInstr) {
  ASSERT_TRUE(parseMIR(TwoBlockMIR));
  MachineFunction &MF = getMF("func");
  SlotIndexes &SI = MFAM.getResult<SlotIndexesAnalysis>(MF);

  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  SlotIndex First = SI.getInstructionIndex(*MBB0.begin());
  SlotIndex Second = SI.getInstructionIndex(*std::next(MBB0.begin()));

  erase(*std::next(MBB0.begin()), SI);

  EXPECT_TRUE(SI.isStaleIndex(Second));
  EXPECT_FALSE(SI.isBlockBoundaryIndex(Second));
  EXPECT_EQ(SI.canonicalizeIndex(Second), First.getRegSlot());
  // Every slot resolves alike, and the dead slot is left free to grow into.
  for (SlotIndex Idx : {Second, Second.getRegSlot(true), Second.getRegSlot(),
                        Second.getDeadSlot()})
    EXPECT_EQ(SI.canonicalizeIndex(Idx), First.getRegSlot());
  EXPECT_LT(SI.canonicalizeIndex(Second), First.getDeadSlot());
}

TEST_F(SlotIndexesTest, RunOfErasedInstrsResolvesToSameIndex) {
  ASSERT_TRUE(parseMIR(TwoBlockMIR));
  MachineFunction &MF = getMF("func");
  SlotIndexes &SI = MFAM.getResult<SlotIndexesAnalysis>(MF);

  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  SlotIndex First = SI.getInstructionIndex(*MBB0.begin());
  SlotIndex Second = SI.getInstructionIndex(*std::next(MBB0.begin()));
  SlotIndex Third = SI.getInstructionIndex(*std::next(MBB0.begin(), 2));

  erase(*std::next(MBB0.begin(), 2), SI);
  erase(*std::next(MBB0.begin()), SI);

  EXPECT_TRUE(SI.isStaleIndex(Second));
  EXPECT_TRUE(SI.isStaleIndex(Third));
  EXPECT_EQ(SI.canonicalizeIndex(Second), First.getRegSlot());
  EXPECT_EQ(SI.canonicalizeIndex(Third), First.getRegSlot());
}

// The block start is shared with the previous block's end index, so the result
// must still report as belonging to the erased instruction's own block.
TEST_F(SlotIndexesTest, ErasedBlockPrefixResolvesToBlockStart) {
  ASSERT_TRUE(parseMIR(TwoBlockMIR));
  MachineFunction &MF = getMF("func");
  SlotIndexes &SI = MFAM.getResult<SlotIndexesAnalysis>(MF);

  MachineBasicBlock &MBB1 = *MF.getBlockNumbered(1);
  SlotIndex Start = SI.getMBBStartIdx(&MBB1);
  SlotIndex FirstIdx = SI.getInstructionIndex(*MBB1.begin());

  erase(*MBB1.begin(), SI);

  SlotIndex Canonical = SI.canonicalizeIndex(FirstIdx);
  EXPECT_EQ(Canonical, Start);
  EXPECT_FALSE(SI.isStaleIndex(Canonical));
  EXPECT_EQ(SI.getMBBFromIndex(Canonical), &MBB1);
  // There is still a slot above it to grow into.
  EXPECT_FALSE(SI.isStaleIndex(Canonical.getNextSlot()));
  EXPECT_GT(Canonical.getNextSlot(), SI.getMBBEndIdx(MF.getBlockNumbered(0)));
}

TEST_F(SlotIndexesTest, ErasedEntryBlockResolvesToZeroIndex) {
  ASSERT_TRUE(parseMIR(TwoBlockMIR));
  MachineFunction &MF = getMF("func");
  SlotIndexes &SI = MFAM.getResult<SlotIndexesAnalysis>(MF);

  MachineBasicBlock &MBB0 = *MF.getBlockNumbered(0);
  SlotIndex Start = SI.getMBBStartIdx(&MBB0);
  SmallVector<SlotIndex> Indexes;
  for (MachineInstr &MI : MBB0)
    Indexes.push_back(SI.getInstructionIndex(MI));

  for (MachineInstr &MI : make_early_inc_range(MBB0))
    erase(MI, SI);

  for (SlotIndex Idx : Indexes) {
    EXPECT_TRUE(SI.isStaleIndex(Idx));
    EXPECT_EQ(SI.canonicalizeIndex(Idx), Start);
  }
}

// One block means one entry in the index -> MBB map, so the upper-bound search
// always lands on its end.
TEST_F(SlotIndexesTest, SingleBlockFunction) {
  ASSERT_TRUE(parseMIR(R"(
---
name: func
tracksRegLiveness: true
body:             |
  bb.0:
    S_NOP 0
    S_ENDPGM 0
...
)"));
  MachineFunction &MF = getMF("func");
  SlotIndexes &SI = MFAM.getResult<SlotIndexesAnalysis>(MF);

  MachineBasicBlock &MBB = MF.front();
  SlotIndex Start = SI.getMBBStartIdx(&MBB);
  SlotIndex Nop = SI.getInstructionIndex(*MBB.begin());
  SlotIndex End = SI.getInstructionIndex(MBB.back());

  EXPECT_TRUE(SI.isBlockBoundaryIndex(Start));
  EXPECT_TRUE(SI.isBlockBoundaryIndex(SI.getMBBEndIdx(&MBB)));

  erase(*MBB.begin(), SI);
  EXPECT_TRUE(SI.isStaleIndex(Nop));
  EXPECT_EQ(SI.canonicalizeIndex(Nop), Start);
  EXPECT_FALSE(SI.isStaleIndex(End));
  EXPECT_TRUE(SI.isBlockBoundaryIndex(SI.getMBBEndIdx(&MBB)));
}

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

/// Leftover entries inflate live range sizes, which is what misled the
/// allocator. Compaction must restore the real def-to-use size.
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
  EXPECT_EQ(LIS.getInterval(Tracked).getSize(), 4u * SlotIndex::InstrDist);

  for (Register Reg : {Register::index2VirtReg(1), Register::index2VirtReg(2),
                       Register::index2VirtReg(3)})
    deleteInstr(*MF.getRegInfo().getVRegDef(Reg), LIS);
  ASSERT_EQ(std::distance(MBB.begin(), MBB.end()), 3L);

  // Deleting does not renumber, so the size is unchanged.
  EXPECT_EQ(LIS.getInterval(Tracked).getSize(), 4u * SlotIndex::InstrDist);

  EXPECT_EQ(compact(LIS), 3u);

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

/// An unreported index still has to compare sanely, since only an asserts
/// build diagnoses it. It takes the index of the surviving entry behind it.
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
    EXPECT_LT(S, SI.getLastIndex());
    // Ordered with the entry behind, but no longer distinct from it.
    EXPECT_FALSE(S < Behind);
    EXPECT_FALSE(Behind < S);
    EXPECT_FALSE(S == Behind);
  }
}

/// No caller can check that every holder reported, so an unreported index is
/// diagnosed at the point of use rather than followed.
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
  // An outside holder that never reports this index.
  const SlotIndex Stale = LIS.getInstructionIndex(Dead).getBaseIndex();
  deleteInstr(Dead, LIS);

  ASSERT_EQ(compact(LIS), 1u);

  EXPECT_DEATH((void)SI.getInstructionFromIndex(Stale),
               "SlotIndex outlived the instruction it pointed at");
}
#endif

} // namespace
