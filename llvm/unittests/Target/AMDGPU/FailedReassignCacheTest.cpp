//===- FailedReassignCacheTest.cpp - FailedReassignCache unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FailedReassignCache.h"
#include "AMDGPUUnitTests.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

constexpr unsigned NumKeepVGPRs = 8;

class FailedReassignCacheTest : public AMDGPUCodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgpu9.00-amd-amdhsa", "", /*FS=*/""); }

  struct Fixture;

  void runWithFixture(unsigned MaxFreedRegs,
                      function_ref<void(Fixture &)> Body);

  /// Leave only the first \p NumKeep members of VGPR_32 allocatable.
  static void limitVGPROrder(MachineFunction &MF, unsigned NumKeep) {
    MachineRegisterInfo &MRI = MF.getRegInfo();
    const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
    MRI.freezeReservedRegs();
    unsigned Kept = 0;
    for (MCRegister Reg : AMDGPU::VGPR_32RegClass) {
      if (Kept < NumKeep) {
        ++Kept;
        continue;
      }
      MRI.reserveReg(Reg, &TRI);
    }
  }

  struct Fixture {
    MachineFunction &MF;
    LiveIntervals &LIS;
    VirtRegMap &VRM;
    LiveRegMatrix &Matrix;
    RegisterClassInfo RCI;
    reassign::Cache Cache;

    Fixture(MachineFunction &MF, LiveIntervals &LIS, VirtRegMap &VRM,
            LiveRegMatrix &Matrix, unsigned MaxFreedRegs = 64)
        : MF(MF), LIS(LIS), VRM(VRM), Matrix(Matrix),
          Cache(LIS, Matrix, MF.getRegInfo(),
                *MF.getSubtarget().getRegisterInfo(), RCI, MaxFreedRegs) {
      limitVGPROrder(MF, NumKeepVGPRs);
      RCI.runOnMachineFunction(MF);

      ArrayRef<MCPhysReg> Order = RCI.getOrder(&AMDGPU::VGPR_32RegClass);
      EXPECT_EQ(Order.size(), NumKeepVGPRs);
    }

    const LiveInterval &LI(unsigned VirtIdx) {
      return LIS.getInterval(Register::index2VirtReg(VirtIdx));
    }

    MCRegister vgpr(unsigned I) const { return MCRegister(AMDGPU::VGPR0 + I); }

    void assign(unsigned VirtIdx, unsigned PhysIdx) {
      Matrix.assign(LI(VirtIdx), vgpr(PhysIdx));
    }

    void unassign(unsigned VirtIdx) { Matrix.unassign(LI(VirtIdx)); }

    /// Insert enough instructions that packIndexes() moves \p Range into a
    /// different slot-index bucket, and pack.
    void shiftAcrossBucket(const LiveInterval &Range) {
      SlotIndexes &Indexes = *LIS.getSlotIndexes();
      SlotIndex Zero = Indexes.getZeroIndex();
      unsigned Before =
          static_cast<unsigned>(Zero.distance(Range.beginIndex())) >>
          reassign::Cache::BucketShift;

      MachineBasicBlock &MBB = MF.front();
      const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
      DebugLoc DL;
      MachineBasicBlock::iterator InsertPt = MBB.begin();
      // One bucket is 1 << BucketShift index units. Packed instructions are
      // SlotIndex::InstrDist apart, so this many new entries advance every
      // later index by one bucket.
      constexpr unsigned NumInsert =
          (1u << reassign::Cache::BucketShift) / SlotIndex::InstrDist;
      for (unsigned I = 0; I < NumInsert; ++I) {
        MachineInstr *MI =
            BuildMI(MBB, InsertPt, DL, TII.get(AMDGPU::S_NOP)).addImm(0);
        LIS.InsertMachineInstrInMaps(*MI);
      }
      Indexes.packIndexes();

      unsigned After =
          static_cast<unsigned>(Zero.distance(Range.beginIndex())) >>
          reassign::Cache::BucketShift;
      EXPECT_NE(Before, After);
    }

    void installListener() {
      Matrix.setUnassignListener(
          [](void *Ctx, const LiveInterval &Unassigned, MCRegister PhysReg,
             bool ClearAll) {
            static_cast<reassign::Cache *>(Ctx)->noteUnassign(
                Unassigned, PhysReg, ClearAll);
          },
          &Cache);
    }
  };

  static constexpr const char *MIR = R"MIR(
---
name: cache_test
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = IMPLICIT_DEF
    %1:vgpr_32 = IMPLICIT_DEF
    %2:vgpr_32 = IMPLICIT_DEF
    %3:vgpr_32 = IMPLICIT_DEF
    %4:vgpr_32 = IMPLICIT_DEF
    %5:vgpr_32 = IMPLICIT_DEF
    %6:vgpr_32 = IMPLICIT_DEF
    %7:vgpr_32 = IMPLICIT_DEF
    S_NOP 0, implicit %0, implicit %1, implicit %2, implicit %3, implicit %4, implicit %5, implicit %6, implicit %7
    S_ENDPGM 0
...
)MIR";
};

void FailedReassignCacheTest::runWithFixture(
    unsigned MaxFreedRegs, function_ref<void(Fixture &)> Body) {
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("cache_test");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  VirtRegMap &VRM = MFAM.getResult<VirtRegMapAnalysis>(MF);
  LiveRegMatrix &Matrix = MFAM.getResult<LiveRegMatrixAnalysis>(MF);
  Fixture F(MF, LIS, VRM, Matrix, MaxFreedRegs);
  Body(F);
}

TEST_F(FailedReassignCacheTest, RememberAndHitFalse) {
  runWithFixture(64, [&](Fixture &F) {
    const LiveInterval &LI = F.LI(0);
    F.Cache.remember(LI, /*HintOnly=*/false);

    EXPECT_TRUE(F.Cache.contains(LI.reg().id()));
    EXPECT_TRUE(F.Cache.freedRegs(LI.reg().id()).empty());
    EXPECT_EQ(F.Cache.reuse(LI, F.vgpr(0)), reassign::Reuse::HitFalse);
    EXPECT_TRUE(F.Cache.contains(LI.reg().id()));
  });
}

TEST_F(FailedReassignCacheTest, ForgetErase) {
  runWithFixture(64, [&](Fixture &F) {
    const LiveInterval &LI = F.LI(0);
    F.Cache.remember(LI, /*HintOnly=*/false);
    F.Cache.erase(LI.reg().id());
    EXPECT_TRUE(F.Cache.empty());
    EXPECT_EQ(F.Cache.reuse(LI, F.vgpr(0)), reassign::Reuse::Miss);
  });
}

TEST_F(FailedReassignCacheTest, NoteUnassignEmptyCacheIsNoOp) {
  runWithFixture(64, [&](Fixture &F) {
    F.assign(1, 1);
    F.Cache.noteUnassign(F.LI(1), F.vgpr(1),
                         /*ClearAllReferencingSegments=*/false);
    EXPECT_TRUE(F.Cache.empty());
  });
}

TEST_F(FailedReassignCacheTest, NoteUnassignRecordsOverlappingFree) {
  runWithFixture(64, [&](Fixture &F) {
    F.installListener();
    for (unsigned I = 1; I < NumKeepVGPRs; ++I)
      F.assign(I, I);

    const LiveInterval &LI = F.LI(0);
    F.Cache.remember(LI, /*HintOnly=*/false);
    EXPECT_EQ(F.Cache.reuse(LI, F.vgpr(0)), reassign::Reuse::HitFalse);

    F.unassign(1);
    ArrayRef<MCPhysReg> Freed = F.Cache.freedRegs(LI.reg().id());
    ASSERT_EQ(Freed.size(), 1u);
    EXPECT_EQ(Freed[0], static_cast<MCPhysReg>(AMDGPU::VGPR1));
  });
}

TEST_F(FailedReassignCacheTest, RecheckTrueAfterOverlappingUnassign) {
  runWithFixture(64, [&](Fixture &F) {
    F.installListener();
    for (unsigned I = 1; I < NumKeepVGPRs; ++I)
      F.assign(I, I);

    const LiveInterval &LI = F.LI(0);
    F.Cache.remember(LI, /*HintOnly=*/false);
    F.unassign(1);

    EXPECT_EQ(F.Cache.reuse(LI, F.vgpr(0)), reassign::Reuse::RecheckTrue);
    EXPECT_FALSE(F.Cache.contains(LI.reg().id()));
  });
}

TEST_F(FailedReassignCacheTest, RecheckFalseKeepsFingerprint) {
  runWithFixture(64, [&](Fixture &F) {
    for (unsigned I = 1; I < NumKeepVGPRs; ++I)
      F.assign(I, I);

    const LiveInterval &LI = F.LI(0);
    F.Cache.remember(LI, /*HintOnly=*/false);

    // Record VGPR1 as freed while it is still occupied, so the recheck fails.
    F.Cache.noteUnassign(F.LI(1), F.vgpr(1),
                         /*ClearAllReferencingSegments=*/false);
    EXPECT_EQ(F.Cache.reuse(LI, F.vgpr(0)), reassign::Reuse::RecheckFalse);
    EXPECT_TRUE(F.Cache.contains(LI.reg().id()));
    EXPECT_TRUE(F.Cache.freedRegs(LI.reg().id()).empty());
    EXPECT_EQ(F.Cache.reuse(LI, F.vgpr(0)), reassign::Reuse::HitFalse);
  });
}

TEST_F(FailedReassignCacheTest, MultiEntryNoteUnassignUpdatesOverlap) {
  runWithFixture(64, [&](Fixture &F) {
    const LiveInterval &A = F.LI(0);
    const LiveInterval &B = F.LI(1);
    F.Cache.remember(A, /*HintOnly=*/false);
    F.Cache.remember(B, /*HintOnly=*/false);

    F.Cache.noteUnassign(A, F.vgpr(2), /*ClearAllReferencingSegments=*/false);
    // A's own failure is dropped so the next query recomputes it. Interior
    // segment bounds are not part of the fingerprint.
    EXPECT_FALSE(F.Cache.contains(A.reg().id()));
    EXPECT_EQ(F.Cache.reuse(A, F.vgpr(2)), reassign::Reuse::Miss);
    ArrayRef<MCPhysReg> FreedB = F.Cache.freedRegs(B.reg().id());
    ASSERT_EQ(FreedB.size(), 1u);
    // Both live in the same block, so the free updates every other overlapping
    // entry.
    EXPECT_EQ(FreedB[0], static_cast<MCPhysReg>(AMDGPU::VGPR2));
  });
}

TEST_F(FailedReassignCacheTest, ClearAllReferencingSegmentsDropsCache) {
  runWithFixture(64, [&](Fixture &F) {
    F.Cache.remember(F.LI(0), /*HintOnly=*/false);
    F.Cache.remember(F.LI(1), /*HintOnly=*/false);
    F.Cache.noteUnassign(F.LI(2), F.vgpr(2),
                         /*ClearAllReferencingSegments=*/true);
    EXPECT_TRUE(F.Cache.empty());
  });
}

TEST_F(FailedReassignCacheTest, HintOnlyMissesOnceFreed) {
  runWithFixture(64, [&](Fixture &F) {
    const LiveInterval &LI = F.LI(0);
    F.Cache.remember(LI, /*HintOnly=*/true);
    // Hint-only is not a hit even before any free: the hint set is not
    // validated against the matrix.
    EXPECT_EQ(F.Cache.reuse(LI, F.vgpr(0)), reassign::Reuse::Miss);
    EXPECT_TRUE(F.Cache.contains(LI.reg().id()));

    F.Cache.noteUnassign(F.LI(1), F.vgpr(1),
                         /*ClearAllReferencingSegments=*/false);
    EXPECT_EQ(F.Cache.reuse(LI, F.vgpr(0)), reassign::Reuse::Miss);
    EXPECT_TRUE(F.Cache.contains(LI.reg().id()));
  });
}

TEST_F(FailedReassignCacheTest, HintOnlyMissesOnNonOverlappingUnassign) {
  runWithFixture(64, [&](Fixture &F) {
    const LiveInterval &LI = F.LI(0);
    F.Cache.remember(LI, /*HintOnly=*/true);

    // A free that ends where LI starts does not overlap, so FreedRegs stays
    // empty. TILE-style hints can still gain a destination from that free.
    SlotIndex Begin = LI.beginIndex();
    SlotIndex Prev = Begin.getPrevIndex();
    ASSERT_TRUE(Prev < Begin);
    LiveInterval Outside(F.LI(1).reg(), 0.0f);
    Outside.addSegment(LiveRange::Segment(Prev, Begin, nullptr));
    F.Cache.noteUnassign(Outside, F.vgpr(1),
                         /*ClearAllReferencingSegments=*/false);

    EXPECT_TRUE(F.Cache.freedRegs(LI.reg().id()).empty());
    EXPECT_EQ(F.Cache.reuse(LI, F.vgpr(0)), reassign::Reuse::Miss);
    EXPECT_TRUE(F.Cache.contains(LI.reg().id()));
  });
}

TEST_F(FailedReassignCacheTest, FingerprintMismatchErases) {
  runWithFixture(64, [&](Fixture &F) {
    const LiveInterval &LI = F.LI(0);
    F.Cache.remember(LI, /*HintOnly=*/false);

    LiveInterval Fake(LI.reg(), 0.0f);
    SlotIndex Begin = LI.beginIndex();
    SlotIndex Mid = Begin.getNextIndex();
    ASSERT_TRUE(Mid < LI.endIndex());
    Fake.addSegment(LiveRange::Segment(Begin, Mid, nullptr));

    EXPECT_EQ(F.Cache.reuse(Fake, F.vgpr(0)), reassign::Reuse::Miss);
    EXPECT_FALSE(F.Cache.contains(LI.reg().id()));
  });
}

TEST_F(FailedReassignCacheTest, SaturatesAfterMaxFreedRegs) {
  // Cap at 2 so saturation is reachable with the 8-VGPR order.
  runWithFixture(2, [&](Fixture &F) {
    const LiveInterval &LI = F.LI(0);
    F.Cache.remember(LI, /*HintOnly=*/false);

    F.Cache.noteUnassign(F.LI(1), F.vgpr(1), false);
    F.Cache.noteUnassign(F.LI(2), F.vgpr(2), false);
    EXPECT_EQ(F.Cache.freedRegs(LI.reg().id()).size(), 2u);
    EXPECT_FALSE(F.Cache.isSaturated(LI.reg().id()));

    F.Cache.noteUnassign(F.LI(3), F.vgpr(3), false);
    EXPECT_TRUE(F.Cache.isSaturated(LI.reg().id()));
    EXPECT_TRUE(F.Cache.freedRegs(LI.reg().id()).empty());
    EXPECT_EQ(F.Cache.reuse(LI, F.vgpr(0)), reassign::Reuse::Miss);
  });
}

TEST_F(FailedReassignCacheTest, DuplicateFreeIsIgnored) {
  runWithFixture(64, [&](Fixture &F) {
    const LiveInterval &LI = F.LI(0);
    F.Cache.remember(LI, /*HintOnly=*/false);
    F.Cache.noteUnassign(F.LI(1), F.vgpr(1), false);
    F.Cache.noteUnassign(F.LI(1), F.vgpr(1), false);
    EXPECT_EQ(F.Cache.freedRegs(LI.reg().id()).size(), 1u);
  });
}

TEST_F(FailedReassignCacheTest, RenumberDropsCache) {
  runWithFixture(64, [&](Fixture &F) {
    const LiveInterval &LI = F.LI(0);
    F.Cache.remember(LI, /*HintOnly=*/false);
    F.shiftAcrossBucket(LI);

    // The fingerprint would still match, but bucket keys are stale. Consulting
    // the cache drops it instead of returning HitFalse.
    EXPECT_EQ(F.Cache.reuse(LI, F.vgpr(0)), reassign::Reuse::Miss);
    EXPECT_TRUE(F.Cache.empty());
  });

  runWithFixture(64, [&](Fixture &F) {
    F.installListener();
    for (unsigned I = 1; I < NumKeepVGPRs; ++I)
      F.assign(I, I);

    const LiveInterval &LI = F.LI(0);
    F.Cache.remember(LI, /*HintOnly=*/false);
    F.shiftAcrossBucket(LI);

    // The unassign searches the new buckets and would miss this entry. Drop
    // the cache instead of leaving a stale failure.
    F.unassign(1);
    EXPECT_TRUE(F.Cache.empty());
    EXPECT_EQ(F.Cache.reuse(LI, F.vgpr(0)), reassign::Reuse::Miss);
  });
}

TEST_F(FailedReassignCacheTest, NestedAdvisorsBothSeeUnassign) {
  runWithFixture(64, [&](Fixture &F) {
    // MLEvictAdvisor constructs the base advisor, which registers a listener,
    // then an embedded DefaultEvictionAdvisor, which registers another. The
    // outer cache must still observe unassigns.
    const TargetRegisterInfo &TRI = *F.MF.getSubtarget().getRegisterInfo();
    reassign::Cache Outer(F.LIS, F.Matrix, F.MF.getRegInfo(), TRI, F.RCI);
    reassign::Cache Inner(F.LIS, F.Matrix, F.MF.getRegInfo(), TRI, F.RCI);
    auto Listen = [](void *Ctx, const LiveInterval &Unassigned,
                     MCRegister PhysReg, bool ClearAll) {
      static_cast<reassign::Cache *>(Ctx)->noteUnassign(Unassigned, PhysReg,
                                                        ClearAll);
    };
    F.Matrix.setUnassignListener(Listen, &Outer);
    F.Matrix.setUnassignListener(Listen, &Inner);

    for (unsigned I = 1; I < NumKeepVGPRs; ++I)
      F.assign(I, I);

    const LiveInterval &LI = F.LI(0);
    Outer.remember(LI, /*HintOnly=*/false);
    Inner.remember(LI, /*HintOnly=*/false);
    F.unassign(1);

    EXPECT_EQ(Outer.reuse(LI, F.vgpr(0)), reassign::Reuse::RecheckTrue);
    EXPECT_EQ(Inner.reuse(LI, F.vgpr(0)), reassign::Reuse::RecheckTrue);

    // Dropping the embedded advisor must leave the outer listener in place.
    F.Matrix.clearUnassignListener(&Inner);
    Outer.remember(LI, /*HintOnly=*/false);
    F.unassign(2);
    EXPECT_EQ(Outer.reuse(LI, F.vgpr(0)), reassign::Reuse::RecheckTrue);

    F.Matrix.clearUnassignListener(&Outer);
  });
}

TEST_F(FailedReassignCacheTest, ClearEmptiesEntriesAndBuckets) {
  runWithFixture(64, [&](Fixture &F) {
    F.Cache.remember(F.LI(0), false);
    F.Cache.remember(F.LI(1), false);
    F.Cache.clear();
    EXPECT_TRUE(F.Cache.empty());
    EXPECT_EQ(F.Cache.reuse(F.LI(0), F.vgpr(0)), reassign::Reuse::Miss);
  });
}

} // namespace
