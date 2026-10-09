//===- AMDGPUEventTrackingTest.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUEventTracking.h"
#include "AMDGPUUnitTests.h"
#include "AMDGPUWaitcntUtils.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::AMDGPU;
using namespace llvm::AMDGPU::eventtracking;

namespace {

static constexpr unsigned CounterLimit = 12;

// These are not accurate, they are simply for testing purposes.
// We do not need to test every single counter accurately, that is the job
// of the IR/MIR tests in tests/CodeGen/AMDGPU. We just need enough here
// to validate that the EventTracker works.
static constexpr std::array<CounterInfo, 4> GFX12CounterInfos = {{
    {LOAD_CNT, HWEvents::VMEM_READ_ACCESS, CounterLimit},
    {DS_CNT, HWEvents::LDS_ACCESS, CounterLimit},
    {EXP_CNT, HWEvents::EXP_GPR_LOCK, CounterLimit},
    {STORE_CNT, HWEvents::VMEM_WRITE_ACCESS | HWEvents::SCRATCH_WRITE_ACCESS,
     CounterLimit},
}};

class AMDGPUGFX12EventTrackingTest : public AMDGPUCodeGenTestBase {
public:
  void SetUp() override {
    setUpImpl("amdgpu12.00-amd-amdhsa", "", "");

    TrackerGetter = [&](const MachineBasicBlock &MBB) -> EventTracker & {
      auto &Entry = Trackers[&MBB];
      if (!Entry)
        Entry = std::make_unique<EventTracker>(MBB, GFX12CounterInfos);
      return *Entry;
    };
  }

  EventTracker &
  visitAll(MachineBasicBlock &MBB,
           function_ref<void(EventTracker &ET)> AfterVisit = nullptr) {
    const GCNSubtarget &ST = MBB.getParent()->getSubtarget<GCNSubtarget>();

    EventTracker &ET = TrackerGetter(MBB);
    ET.enterBlock(TrackerGetter);
    for (MachineInstr &MI : MBB) {
      HWEvents Events =
          getEventsFor(MI, ST, /*IsExpertMode=*/false, /*TgSplit=*/false);
      for (HWEvents SingleEv : Events) {
        ET.record(MI, SingleHWEvent::encode(SingleEv));
      }
    }
    if (AfterVisit)
      AfterVisit(ET);
    return ET;
  }

  std::function<EventTracker &(const MachineBasicBlock &)> TrackerGetter;
  DenseMap<const MachineBasicBlock *, std::unique_ptr<EventTracker>> Trackers;
};

namespace {

/// Provides some helpers to declaratively check the state of the EventTracker
/// for one counter. This provides helpers to check the general counter state
/// (value, etc) but also allows iterating over the timeline from the oldest to
/// the youngest element.
///
/// This makes the actual test cases clearer.
struct TrackerRecordsChecker {
  TrackerRecordsChecker(EventTracker &ET, InstCounterType T)
      : ET(ET), T(T), Records(ET.getTimeline(T)) {}

  unsigned getCount() { return ET.count(T); }

  bool empty() { return Records.empty(); }

  // Has no records and count is zero
  bool unused() { return empty() && !getCount(); }

  const EventTrackerRecord &cur() { return Records[CurElt]; }

  /// Move to the next record.
  /// \returns true on success, false if the end has been reached.
  bool next() {
    ++CurElt;
    if (CurElt >= Records.size())
      return false;
    return true;
  }

  EventTracker &ET;
  InstCounterType T;
  ArrayRef<EventTrackerRecord> Records;
  unsigned CurElt = 0;
};

} // namespace

/// Check trivial straight line counting.
TEST_F(AMDGPUGFX12EventTrackingTest, BasicTimeline) {
  StringRef MIR = R"(
name:            BasicTimeline
body:             |
  bb.0:

    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr4 = GLOBAL_LOAD_DWORD $vgpr4_vgpr5, 0, 0, implicit $exec
    $vgpr0 = DS_READ_B32_gfx9 $vgpr1, 0, 0, implicit $exec
    GLOBAL_STORE_DWORDX2 $vgpr1_vgpr2, $vgpr3_vgpr4, 0, 0, implicit $exec
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("BasicTimeline");
  MachineBasicBlock &BB0 = *MF.getBlockNumbered(0);

  auto &ET = visitAll(BB0);

  auto LoadCnt = TrackerRecordsChecker(ET, AMDGPU::LOAD_CNT);
  EXPECT_EQ(LoadCnt.getCount(), 1u);
  EXPECT_FALSE(LoadCnt.empty());
  EXPECT_EQ(LoadCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_LOAD_DWORD);
  EXPECT_EQ(LoadCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(LoadCnt.next());

  auto DsCnt = TrackerRecordsChecker(ET, AMDGPU::DS_CNT);
  EXPECT_EQ(DsCnt.getCount(), 1u);
  EXPECT_FALSE(DsCnt.empty());
  EXPECT_EQ(DsCnt.cur().getMI()->getOpcode(), AMDGPU::DS_READ_B32_gfx9);
  EXPECT_EQ(DsCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(DsCnt.next());

  auto ExpCnt = TrackerRecordsChecker(ET, AMDGPU::EXP_CNT);
  EXPECT_TRUE(ExpCnt.unused());

  auto StoreCnt = TrackerRecordsChecker(ET, AMDGPU::STORE_CNT);
  EXPECT_EQ(StoreCnt.getCount(), 2u);
  EXPECT_FALSE(StoreCnt.empty());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORD);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 1u);
  EXPECT_TRUE(StoreCnt.next());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORDX2);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(StoreCnt.next());
}

/// Check cases where a block loops back on itself.
TEST_F(AMDGPUGFX12EventTrackingTest, SelfPredecessor) {
  StringRef MIR = R"(
name:            BasicTimeline
body:             |
  bb.0:

    $vgpr4 = GLOBAL_LOAD_DWORD $vgpr4_vgpr5, 0, 0, implicit $exec
    $vgpr0 = DS_READ_B32_gfx9 $vgpr1, 0, 0, implicit $exec
    GLOBAL_STORE_DWORDX2 $vgpr1_vgpr2, $vgpr3_vgpr4, 0, 0, implicit $exec
    S_CBRANCH_SCC1 %bb.0, implicit $scc
    S_BRANCH %bb.1

  bb.1:
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("BasicTimeline");
  MachineBasicBlock &BB0 = *MF.getBlockNumbered(0);

  // Iterate twice
  visitAll(BB0);
  auto &ET = visitAll(BB0);

  auto LoadCnt = TrackerRecordsChecker(ET, AMDGPU::LOAD_CNT);
  EXPECT_EQ(LoadCnt.getCount(), 2u);
  EXPECT_FALSE(LoadCnt.empty());
  EXPECT_EQ(LoadCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_LOAD_DWORD);
  EXPECT_EQ(LoadCnt.cur().getHeight(), 1u);
  EXPECT_TRUE(LoadCnt.next());
  EXPECT_EQ(LoadCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_LOAD_DWORD);
  EXPECT_EQ(LoadCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(LoadCnt.next());

  auto DsCnt = TrackerRecordsChecker(ET, AMDGPU::DS_CNT);
  EXPECT_EQ(DsCnt.getCount(), 2u);
  EXPECT_FALSE(DsCnt.empty());
  EXPECT_EQ(DsCnt.cur().getMI()->getOpcode(), AMDGPU::DS_READ_B32_gfx9);
  EXPECT_EQ(DsCnt.cur().getHeight(), 1u);
  EXPECT_TRUE(DsCnt.next());
  EXPECT_EQ(DsCnt.cur().getMI()->getOpcode(), AMDGPU::DS_READ_B32_gfx9);
  EXPECT_EQ(DsCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(DsCnt.next());

  auto ExpCnt = TrackerRecordsChecker(ET, AMDGPU::EXP_CNT);
  EXPECT_TRUE(ExpCnt.unused());

  auto StoreCnt = TrackerRecordsChecker(ET, AMDGPU::STORE_CNT);
  EXPECT_EQ(StoreCnt.getCount(), 2u);
  EXPECT_FALSE(StoreCnt.empty());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORDX2);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 1u);
  EXPECT_TRUE(StoreCnt.next());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORDX2);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(StoreCnt.next());
}

/// Check all events carry into the next incoming block.
TEST_F(AMDGPUGFX12EventTrackingTest, SingleIncomingBlock) {
  StringRef MIR = R"(
name:            SingleIncomingBlock
body:             |
  bb.0:
    successors: %bb.1

    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    $vgpr4 = GLOBAL_LOAD_DWORD $vgpr4_vgpr5, 0, 0, implicit $exec
    $vgpr0 = DS_READ_B32_gfx9 $vgpr1, 0, 0, implicit $exec
    GLOBAL_STORE_DWORDX2 $vgpr1_vgpr2, $vgpr3_vgpr4, 0, 0, implicit $exec

  bb.1:

    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("SingleIncomingBlock");
  MachineBasicBlock &BB0 = *MF.getBlockNumbered(0);
  MachineBasicBlock &BB1 = *MF.getBlockNumbered(1);

  visitAll(BB0);
  auto &ET = visitAll(BB1);

  auto LoadCnt = TrackerRecordsChecker(ET, AMDGPU::LOAD_CNT);
  EXPECT_EQ(LoadCnt.getCount(), 1u);
  EXPECT_FALSE(LoadCnt.empty());
  EXPECT_EQ(LoadCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_LOAD_DWORD);
  EXPECT_EQ(LoadCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(LoadCnt.next());

  auto DsCnt = TrackerRecordsChecker(ET, AMDGPU::DS_CNT);
  EXPECT_EQ(DsCnt.getCount(), 1u);
  EXPECT_FALSE(DsCnt.empty());
  EXPECT_EQ(DsCnt.cur().getMI()->getOpcode(), AMDGPU::DS_READ_B32_gfx9);
  EXPECT_EQ(DsCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(DsCnt.next());

  auto ExpCnt = TrackerRecordsChecker(ET, AMDGPU::EXP_CNT);
  EXPECT_TRUE(ExpCnt.unused());

  auto StoreCnt = TrackerRecordsChecker(ET, AMDGPU::STORE_CNT);
  EXPECT_EQ(StoreCnt.getCount(), 2u);
  EXPECT_FALSE(StoreCnt.empty());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORD);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 1u);
  EXPECT_TRUE(StoreCnt.next());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORDX2);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(StoreCnt.next());
}

/// Check a simple merge where all events differ in each incoming block.
TEST_F(AMDGPUGFX12EventTrackingTest, SimpleDisjointMerge) {
  StringRef MIR = R"(
name:            SimpleDisjointMerge
body:             |
  bb.0:
    successors: %bb.2

    $vgpr0 = DS_READ_B32_gfx9 $vgpr1, 0, 0, implicit $exec
    $vgpr4 = GLOBAL_LOAD_DWORD $vgpr4_vgpr5, 0, 0, implicit $exec
    S_BRANCH %bb.2

  bb.1:
    successors: %bb.2
    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_STORE_DWORDX2 $vgpr1_vgpr2, $vgpr3_vgpr4, 0, 0, implicit $exec
    S_BRANCH %bb.2

  bb.2:
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("SimpleDisjointMerge");
  MachineBasicBlock &BB0 = *MF.getBlockNumbered(0);
  MachineBasicBlock &BB1 = *MF.getBlockNumbered(1);
  MachineBasicBlock &BB2 = *MF.getBlockNumbered(2);

  visitAll(BB0);
  visitAll(BB1);
  auto &ET = visitAll(BB2);

  auto LoadCnt = TrackerRecordsChecker(ET, AMDGPU::LOAD_CNT);
  EXPECT_EQ(LoadCnt.getCount(), 1u);
  EXPECT_FALSE(LoadCnt.empty());
  EXPECT_EQ(LoadCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_LOAD_DWORD);
  EXPECT_EQ(LoadCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(LoadCnt.next());

  auto DsCnt = TrackerRecordsChecker(ET, AMDGPU::DS_CNT);
  EXPECT_EQ(DsCnt.getCount(), 1u);
  EXPECT_FALSE(DsCnt.empty());
  EXPECT_EQ(DsCnt.cur().getMI()->getOpcode(), AMDGPU::DS_READ_B32_gfx9);
  EXPECT_EQ(DsCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(DsCnt.next());

  auto ExpCnt = TrackerRecordsChecker(ET, AMDGPU::EXP_CNT);
  EXPECT_TRUE(ExpCnt.unused());

  auto StoreCnt = TrackerRecordsChecker(ET, AMDGPU::STORE_CNT);
  EXPECT_EQ(StoreCnt.getCount(), 2u);
  EXPECT_FALSE(StoreCnt.empty());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORD);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 1u);
  EXPECT_TRUE(StoreCnt.next());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORDX2);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(StoreCnt.next());
}

/// Merge with divergent events in a counter.
TEST_F(AMDGPUGFX12EventTrackingTest, DivergenceMerge) {
  StringRef MIR = R"(
name:            DivergenceMerge
body:             |
  bb.0:
    successors: %bb.2

    $vgpr0 = DS_READ_B32_gfx9 $vgpr1, 0, 0, implicit $exec
    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    S_BRANCH %bb.2

  bb.1:
    successors: %bb.2
    $vgpr4 = GLOBAL_LOAD_DWORD $vgpr4_vgpr5, 0, 0, implicit $exec
    GLOBAL_STORE_DWORDX2 $vgpr1_vgpr2, $vgpr3_vgpr4, 0, 0, implicit $exec
    S_BRANCH %bb.2

  bb.2:
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("DivergenceMerge");
  MachineBasicBlock &BB0 = *MF.getBlockNumbered(0);
  MachineBasicBlock &BB1 = *MF.getBlockNumbered(1);
  MachineBasicBlock &BB2 = *MF.getBlockNumbered(2);

  visitAll(BB0);
  visitAll(BB1);
  auto &ET = visitAll(BB2);

  auto LoadCnt = TrackerRecordsChecker(ET, AMDGPU::LOAD_CNT);
  EXPECT_EQ(LoadCnt.getCount(), 1u);
  EXPECT_FALSE(LoadCnt.empty());
  EXPECT_EQ(LoadCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_LOAD_DWORD);
  EXPECT_EQ(LoadCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(LoadCnt.next());

  auto DsCnt = TrackerRecordsChecker(ET, AMDGPU::DS_CNT);
  EXPECT_EQ(DsCnt.getCount(), 1u);
  EXPECT_FALSE(DsCnt.empty());
  EXPECT_EQ(DsCnt.cur().getMI()->getOpcode(), AMDGPU::DS_READ_B32_gfx9);
  EXPECT_EQ(DsCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(DsCnt.next());

  auto ExpCnt = TrackerRecordsChecker(ET, AMDGPU::EXP_CNT);
  EXPECT_TRUE(ExpCnt.unused());

  auto StoreCnt = TrackerRecordsChecker(ET, AMDGPU::STORE_CNT);
  // Count is 1 because we have 1 event max across all predecessors.
  EXPECT_EQ(StoreCnt.getCount(), 1u);
  EXPECT_FALSE(StoreCnt.empty());
  // First successor has GLOBAL_STORE_DWORD at height 0
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORD);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 0u);
  EXPECT_TRUE(StoreCnt.next());
  // Second successor has GLOBAL_STORE_DWORDX2 at height 0 too
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORDX2);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(StoreCnt.next());
}

/// Basic diamond CFG, the store is carried all the way into bb3 and uniqued
/// again so only 1 instance of the record is present in bb3.
TEST_F(AMDGPUGFX12EventTrackingTest, BasicDiamond) {
  StringRef MIR = R"(
name:            BasicDiamond
body:             |
  bb.0:
    successors: %bb.1, %bb.2

    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    S_CBRANCH_SCC1 %bb.1, implicit $scc
    S_BRANCH %bb.2

  bb.1:
    successors: %bb.3
    $vgpr0 = DS_READ_B32_gfx9 $vgpr1, 0, 0, implicit $exec
    S_BRANCH %bb.3

  bb.2:
    successors: %bb.3
    $vgpr4 = GLOBAL_LOAD_DWORD $vgpr4_vgpr5, 0, 0, implicit $exec
    S_BRANCH %bb.3

  bb.3:
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("BasicDiamond");
  MachineBasicBlock &BB0 = *MF.getBlockNumbered(0);
  MachineBasicBlock &BB1 = *MF.getBlockNumbered(1);
  MachineBasicBlock &BB2 = *MF.getBlockNumbered(2);
  MachineBasicBlock &BB3 = *MF.getBlockNumbered(3);

  visitAll(BB0);
  visitAll(BB1);
  visitAll(BB2);
  auto &ET = visitAll(BB3);

  auto LoadCnt = TrackerRecordsChecker(ET, AMDGPU::LOAD_CNT);
  EXPECT_EQ(LoadCnt.getCount(), 1u);
  EXPECT_FALSE(LoadCnt.empty());
  EXPECT_EQ(LoadCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_LOAD_DWORD);
  EXPECT_EQ(LoadCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(LoadCnt.next());

  auto DsCnt = TrackerRecordsChecker(ET, AMDGPU::DS_CNT);
  EXPECT_EQ(DsCnt.getCount(), 1u);
  EXPECT_FALSE(DsCnt.empty());
  EXPECT_EQ(DsCnt.cur().getMI()->getOpcode(), AMDGPU::DS_READ_B32_gfx9);
  EXPECT_EQ(DsCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(DsCnt.next());

  auto ExpCnt = TrackerRecordsChecker(ET, AMDGPU::EXP_CNT);
  EXPECT_TRUE(ExpCnt.unused());

  auto StoreCnt = TrackerRecordsChecker(ET, AMDGPU::STORE_CNT);
  EXPECT_EQ(StoreCnt.getCount(), 1u);
  EXPECT_FALSE(StoreCnt.empty());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORD);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(StoreCnt.next());
}

/// Assymetrical diamond
///   - bb0 has two stores
///   - bb1 adds another store without any waits.
///   - bb2 adds a store and waits on the two stores from bb0 afterwards
///
/// The timeline will be:
///   - Stores from bb0 exist at 1/2
///   - The added store from bb1 exist at height 0
///   - The added store from bb2 exist at height 0
TEST_F(AMDGPUGFX12EventTrackingTest, AssymetricalDiamond) {
  StringRef MIR = R"(
name:            AssymetricalDiamond
body:             |
  bb.0:
    successors: %bb.1, %bb.2

    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_STORE_DWORDX2 $vgpr1_vgpr2, $vgpr3_vgpr4, 0, 0, implicit $exec
    S_CBRANCH_SCC1 %bb.1, implicit $scc
    S_BRANCH %bb.2

  bb.1:
    successors: %bb.3
    GLOBAL_STORE_DWORD $vgpr1_vgpr2, $vgpr3, 0, 0, implicit $exec
    S_BRANCH %bb.3

  bb.2:
    successors: %bb.3
    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    S_BRANCH %bb.3

  bb.3:
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("AssymetricalDiamond");
  MachineBasicBlock &BB0 = *MF.getBlockNumbered(0);
  MachineBasicBlock &BB1 = *MF.getBlockNumbered(1);
  MachineBasicBlock &BB2 = *MF.getBlockNumbered(2);
  MachineBasicBlock &BB3 = *MF.getBlockNumbered(3);

  visitAll(BB0);
  visitAll(BB1);
  visitAll(BB2,
           /*AfterVisit=*/[&](EventTracker &ET) { ET.drain(STORE_CNT, 1); });
  auto &ET = visitAll(BB3);

  auto LoadCnt = TrackerRecordsChecker(ET, AMDGPU::LOAD_CNT);
  EXPECT_TRUE(LoadCnt.unused());

  auto DsCnt = TrackerRecordsChecker(ET, AMDGPU::DS_CNT);
  EXPECT_TRUE(DsCnt.unused());

  auto ExpCnt = TrackerRecordsChecker(ET, AMDGPU::EXP_CNT);
  EXPECT_TRUE(ExpCnt.unused());

  auto StoreCnt = TrackerRecordsChecker(ET, AMDGPU::STORE_CNT);

  EXPECT_EQ(StoreCnt.getCount(), 3u);
  EXPECT_FALSE(StoreCnt.empty());

  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORD);
  EXPECT_EQ(StoreCnt.cur().getMI()->getParent(), &BB0);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 2u);
  EXPECT_TRUE(StoreCnt.next());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORDX2);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 1u);
  EXPECT_TRUE(StoreCnt.next());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORD);
  EXPECT_EQ(StoreCnt.cur().getMI()->getParent(), &BB1);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 0u);
  EXPECT_TRUE(StoreCnt.next());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORD);
  EXPECT_EQ(StoreCnt.cur().getMI()->getParent(), &BB2);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(StoreCnt.next());
}

/// Assymetrical diamond
///   - bb0 has one store
///   - bb1 just falls through
///   - bb2 adds 1 more store.
///
/// The timeline will showcase how the value of a counter can be different from
/// the content of the timeline.
///   - The timeline will have all records at a height of 0.
///   - The counter will still be at 2.
TEST_F(AMDGPUGFX12EventTrackingTest, AssymetricalDiamond2) {
  StringRef MIR = R"(
name:            AssymetricalDiamond2
body:             |
  bb.0:
    successors: %bb.1, %bb.2

    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    S_CBRANCH_SCC1 %bb.1, implicit $scc
    S_BRANCH %bb.2

  bb.1:
    successors: %bb.3
    S_BRANCH %bb.3

  bb.2:
    successors: %bb.3
    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    S_BRANCH %bb.3

  bb.3:
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("AssymetricalDiamond2");
  MachineBasicBlock &BB0 = *MF.getBlockNumbered(0);
  MachineBasicBlock &BB1 = *MF.getBlockNumbered(1);
  MachineBasicBlock &BB2 = *MF.getBlockNumbered(2);
  MachineBasicBlock &BB3 = *MF.getBlockNumbered(3);

  visitAll(BB0);
  visitAll(BB1);
  visitAll(BB2);

  auto &ET = visitAll(BB3);

  auto LoadCnt = TrackerRecordsChecker(ET, AMDGPU::LOAD_CNT);
  EXPECT_TRUE(LoadCnt.unused());

  auto DsCnt = TrackerRecordsChecker(ET, AMDGPU::DS_CNT);
  EXPECT_TRUE(DsCnt.unused());

  auto ExpCnt = TrackerRecordsChecker(ET, AMDGPU::EXP_CNT);
  EXPECT_TRUE(ExpCnt.unused());

  auto StoreCnt = TrackerRecordsChecker(ET, AMDGPU::STORE_CNT);

  EXPECT_EQ(StoreCnt.getCount(), 2u);
  EXPECT_FALSE(StoreCnt.empty());

  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORD);
  EXPECT_EQ(StoreCnt.cur().getMI()->getParent(), &BB0);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 0u);
  EXPECT_TRUE(StoreCnt.next());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORD);
  EXPECT_EQ(StoreCnt.cur().getMI()->getParent(), &BB2);
  EXPECT_EQ(StoreCnt.cur().getHeight(), 0u);
  EXPECT_FALSE(StoreCnt.next());
}

} // namespace
