//===- SlotIndexesTest.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/SlotIndexes.h"
#include "CodeGenTestBase.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
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

} // namespace
