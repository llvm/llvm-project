//===- InlineSpillerTest.cpp - Tests for inline spilling ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CodeGenTestBase.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Config/Targets.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

namespace {

class InlineSpillerTest : public CodeGenTestBase {
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

  // Keep the original interval live over its children, as it is during
  // allocation. Two real uses keep each source out of snippet collection.
  void testSpill(
      StringRef Body, unsigned NewStores,
      function_ref<void(MachineFunction &, LiveIntervals &)> Prepare =
          [](MachineFunction &, LiveIntervals &) {}) {
    std::string MIRString = (Twine(R"MIR(
---
name: func
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
  stackPtrOffsetReg: '$sgpr32'
registers:
  - { id: 0, class: sgpr_128 }
  - { id: 1, class: sgpr_128, split-from: '%0' }
  - { id: 2, class: sgpr_128, split-from: '%0' }
  - { id: 3, class: sgpr_128, split-from: '%0' }
stack:
  - { id: 0, type: spill-slot, size: 16, alignment: 4 }
  - { id: 1, type: spill-slot, size: 16, alignment: 4 }
body: |
)MIR") + Body + "...\n")
                                .str();
    ASSERT_TRUE(parseMIR(MIRString));
    MachineFunction &MF = getMF("func");
    LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
    LiveStacks &LSS = MFAM.getResult<LiveStacksAnalysis>(MF);
    VirtRegMap &VRM = MFAM.getResult<VirtRegMapAnalysis>(MF);
    MachineDominatorTree &MDT =
        MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
    auto &MBFI = MFAM.getResult<MachineBlockFrequencyAnalysis>(MF);
    auto &Loops = MFAM.getResult<MachineLoopAnalysis>(MF);
    VirtRegAuxInfo VRAI(MF, LIS, VRM, Loops, MBFI);

    Register Original = Register::index2VirtReg(0);
    VRM.assignVirt2StackSlot(Original, 0);
    LiveInterval &StackLI =
        LSS.getOrCreateInterval(0, MF.getRegInfo().getRegClass(Original));
    StackLI.getNextValue(SlotIndex(), LSS.getVNInfoAllocator());
    StackLI.MergeSegmentsInAsValue(LIS.getInterval(Original),
                                   StackLI.getValNumInfo(0));
    Prepare(MF, LIS);

    auto CountAccesses = [&](bool Stores) {
      unsigned Count = 0;
      for (const MachineBasicBlock &MBB : MF)
        for (const MachineInstr &MI : MBB) {
          // Count by slot-0 memory operands rather than isStoreToStackSlot so
          // partial-def spills still count: a target may lower them to a pseudo
          // (e.g. SI_SPILL_S32_SAVE_PARTIAL) that isStoreToStackSlot does not
          // report as a complete store. Such a partial store also carries a
          // read-back load operand; exclude it from the reload count.
          bool St = false, Ld = false;
          for (const MachineMemOperand *MMO : MI.memoperands())
            if (auto *FS = dyn_cast_or_null<FixedStackPseudoSourceValue>(
                    MMO->getPseudoValue()))
              if (FS->getFrameIndex() == 0) {
                St |= MMO->isStore();
                Ld |= MMO->isLoad();
              }
          if (Stores ? St : (Ld && !St))
            ++Count;
        }
      return Count;
    };
    unsigned OldStores = CountAccesses(true);
    unsigned OldReloads = CountAccesses(false);
    std::unique_ptr<Spiller> S(
        createInlineSpiller({LIS, LSS, MDT, MBFI}, MF, VRM, VRAI));
    SmallVector<Register> NewRegs;
    LiveRangeEdit Edit(&LIS.getInterval(Register::index2VirtReg(3)), NewRegs,
                       MF, LIS, &VRM);
    S->spill(Edit);

    // Deliberately check before postOptimization: whole-value spill hoisting
    // can otherwise hide a redundant store introduced by the split copy.
    EXPECT_EQ(CountAccesses(true), OldStores + NewStores);
    // Both real uses of the spilled child still need their values. Eliding its
    // incoming copy must not eliminate the reloads supplying those consumers.
    EXPECT_EQ(CountAccesses(false), OldReloads + 2);
    if (!NewStores)
      EXPECT_EQ(NewRegs.size(), 2u);
    EXPECT_TRUE(MF.verify());
  }
};

TEST_F(InlineSpillerTest, ReloadThroughNarrowingSplitCopies) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    SI_SPILL_S128_SAVE %0, %stack.0, implicit $exec, implicit $sgpr32 :: (store (s128) into %stack.0, align 4, addrspace 5)
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    KILL %1
    KILL %1
    undef %2.sub2_sub3 = lr-split COPY %1.sub2_sub3
    KILL %2.sub3
    KILL %2.sub3
    undef %3.sub2 = lr-split COPY %2.sub2
    KILL %3.sub2
    KILL %3.sub2
    KILL %0
    S_ENDPGM 0
)MIR",
            0);
}

TEST_F(InlineSpillerTest, FullWidthSplitCopy) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    KILL %1
    KILL %1
    %3 = lr-split COPY %1
    KILL %3
    KILL %3
    KILL %0
    S_ENDPGM 0
)MIR",
            0);
}

TEST_F(InlineSpillerTest, ChangedOtherLane) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    %1.sub0 = S_MOV_B32 123
    KILL %1
    KILL %1
    undef %3.sub2_sub3 = lr-split COPY %1.sub2_sub3
    KILL %3.sub2_sub3
    KILL %3.sub2_sub3
    KILL %0
    S_ENDPGM 0
)MIR",
            0);
}

TEST_F(InlineSpillerTest, ChangedCopiedLane) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    %1.sub2 = S_ADD_U32 %1.sub2, 1, implicit-def dead $scc
    KILL %1
    KILL %1
    undef %3.sub2_sub3 = lr-split COPY %1.sub2_sub3
    KILL %3.sub2_sub3
    KILL %3.sub2_sub3
    KILL %0
    S_ENDPGM 0
)MIR",
            1);
}

TEST_F(InlineSpillerTest, ReloadFromDifferentSlot) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.1, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.1, align 4, addrspace 5)
    KILL %1
    KILL %1
    undef %3.sub2 = lr-split COPY %1.sub2
    KILL %3.sub2
    KILL %3.sub2
    KILL %0
    S_ENDPGM 0
)MIR",
            1);
}

TEST_F(InlineSpillerTest, InterveningStoreToSameSlot) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    KILL %1
    KILL %1
    SI_SPILL_S128_SAVE %0, %stack.0, implicit $exec, implicit $sgpr32 :: (store (s128) into %stack.0, align 4, addrspace 5)
    undef %3.sub2 = lr-split COPY %1.sub2
    KILL %3.sub2
    KILL %3.sub2
    KILL %0
    S_ENDPGM 0
)MIR",
            1);
}

TEST_F(InlineSpillerTest, InterveningStoreToDifferentSlot) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    KILL %1
    KILL %1
    SI_SPILL_S128_SAVE %0, %stack.1, implicit $exec, implicit $sgpr32 :: (store (s128) into %stack.1, align 4, addrspace 5)
    undef %3.sub2 = lr-split COPY %1.sub2
    KILL %3.sub2
    KILL %3.sub2
    KILL %0
    S_ENDPGM 0
)MIR",
            0);
}

TEST_F(InlineSpillerTest, OrdinaryCopy) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    KILL %1
    KILL %1
    undef %3.sub2 = COPY %1.sub2
    KILL %3.sub2
    KILL %3.sub2
    KILL %0
    S_ENDPGM 0
)MIR",
            1);
}

TEST_F(InlineSpillerTest, ReloadInPredecessor) {
  testSpill(R"MIR(
  bb.0:
    successors: %bb.1
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    KILL %1
    KILL %1
    S_BRANCH %bb.1
  bb.1:
    undef %3.sub2 = lr-split COPY %1.sub2
    KILL %3.sub2
    KILL %3.sub2
    KILL %0
    S_ENDPGM 0
)MIR",
            1);
}

TEST_F(InlineSpillerTest, OriginalMainRangePhi) {
  testSpill(R"MIR(
  bb.0:
    successors: %bb.1
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    SI_SPILL_S128_SAVE %0, %stack.0, implicit $exec, implicit $sgpr32 :: (store (s128) into %stack.0, align 4, addrspace 5)
    S_BRANCH %bb.1
  bb.1:
    successors: %bb.1, %bb.2
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    KILL %1
    KILL %1
    undef %2.sub2_sub3 = lr-split COPY %1.sub2_sub3
    KILL %2.sub3
    KILL %2.sub3
    undef %3.sub2 = lr-split COPY %2.sub2
    KILL %3.sub2
    KILL %3.sub2
    KILL implicit-def $scc
    S_CBRANCH_SCC1 %bb.1, implicit $scc
    S_BRANCH %bb.2
  bb.2:
    KILL %0
    S_ENDPGM 0
)MIR",
            0, [](MachineFunction &MF, LiveIntervals &LIS) {
              // Coalescing can retain this redundant main-range PHI after all
              // lane definitions have moved to the preheader. MIR parsing
              // normalizes it away.
              LiveInterval &LI = LIS.getInterval(Register::index2VirtReg(0));
              SlotIndex Header = LIS.getMBBStartIdx(&*std::next(MF.begin()));
              auto I = LI.find(Header);
              ASSERT_NE(I, LI.end());
              SlotIndex End = I->end;
              VNInfo *Phi = LI.getNextValue(Header, LIS.getVNInfoAllocator());
              LI.removeSegment(Header, End);
              LI.addSegment({Header, End, Phi});
            });
}

TEST_F(InlineSpillerTest, UnknownMemoryClobber) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    KILL %1
    KILL %1
    INLINEASM &"", 17
    undef %3.sub2 = lr-split COPY %1.sub2
    KILL %3.sub2
    KILL %3.sub2
    KILL %0
    S_ENDPGM 0
)MIR",
            1);
}

TEST_F(InlineSpillerTest, RemappedLanes) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    KILL %1
    KILL %1
    undef %3.sub2 = lr-split COPY %1.sub3
    KILL %3.sub2
    KILL %3.sub2
    KILL %0
    S_ENDPGM 0
)MIR",
            1);
}

TEST_F(InlineSpillerTest, CopyOfAnotherOriginal) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %4:sgpr_128 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    KILL %4
    KILL %4
    undef %3.sub2 = lr-split COPY %4.sub2
    KILL %3.sub2
    KILL %3.sub2
    KILL %0
    S_ENDPGM 0
)MIR",
            1);
}

TEST_F(InlineSpillerTest, LanesFromMultipleReloads) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    undef %2.sub2 = lr-split COPY %1.sub2
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    %2.sub3 = lr-split COPY %1.sub3
    KILL %2.sub2_sub3
    KILL %2.sub2_sub3
    undef %3.sub2_sub3 = lr-split COPY %2.sub2_sub3
    KILL %3.sub2_sub3
    KILL %3.sub2_sub3
    KILL %0
    S_ENDPGM 0
)MIR",
            0);
}

TEST_F(InlineSpillerTest, ClobberBetweenLaneReloads) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    undef %2.sub2 = lr-split COPY %1.sub2
    SI_SPILL_S128_SAVE %0, %stack.0, implicit $exec, implicit $sgpr32 :: (store (s128) into %stack.0, align 4, addrspace 5)
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    %2.sub3 = lr-split COPY %1.sub3
    KILL %2.sub2_sub3
    KILL %2.sub2_sub3
    undef %3.sub2_sub3 = lr-split COPY %2.sub2_sub3
    KILL %3.sub2_sub3
    KILL %3.sub2_sub3
    KILL %0
    S_ENDPGM 0
)MIR",
            2);
}

TEST_F(InlineSpillerTest, InterveningCall) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    KILL %1
    KILL %1
    dead $sgpr30_sgpr31 = SI_CALL undef $sgpr4_sgpr5, 0, csr_amdgpu
    undef %3.sub2 = lr-split COPY %1.sub2
    KILL %3.sub2
    KILL %3.sub2
    KILL %0
    S_ENDPGM 0
)MIR",
            1);
}

TEST_F(InlineSpillerTest, PreservingDestination) {
  testSpill(R"MIR(
  bb.0:
    liveins: $sgpr0_sgpr1_sgpr2_sgpr3
    %0 = COPY $sgpr0_sgpr1_sgpr2_sgpr3
    %1 = SI_SPILL_S128_RESTORE %stack.0, implicit $exec, implicit $sgpr32 :: (load (s128) from %stack.0, align 4, addrspace 5)
    KILL %1
    KILL %1
    undef %3.sub0_sub1 = COPY %1.sub0_sub1
    %3.sub2 = lr-split COPY %1.sub2
    KILL %3.sub2
    KILL %3.sub0_sub1
    KILL %0
    S_ENDPGM 0
)MIR",
            3);
}

} // namespace
