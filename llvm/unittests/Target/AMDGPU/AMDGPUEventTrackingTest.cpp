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
  EXPECT_FALSE(LoadCnt.next());

  auto DsCnt = TrackerRecordsChecker(ET, AMDGPU::DS_CNT);
  EXPECT_EQ(DsCnt.getCount(), 1u);
  EXPECT_FALSE(DsCnt.empty());
  EXPECT_EQ(DsCnt.cur().getMI()->getOpcode(), AMDGPU::DS_READ_B32_gfx9);
  EXPECT_FALSE(DsCnt.next());

  auto ExpCnt = TrackerRecordsChecker(ET, AMDGPU::EXP_CNT);
  EXPECT_TRUE(ExpCnt.unused());

  auto StoreCnt = TrackerRecordsChecker(ET, AMDGPU::STORE_CNT);
  EXPECT_EQ(StoreCnt.getCount(), 2u);
  EXPECT_FALSE(StoreCnt.empty());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORD);
  EXPECT_TRUE(StoreCnt.next());
  EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORDX2);
  EXPECT_FALSE(StoreCnt.next());
}

/// Check partial and full drains of a counter.
TEST_F(AMDGPUGFX12EventTrackingTest, Drain) {
  StringRef MIR = R"(
name:            Drain
body:             |
  bb.0:

    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    GLOBAL_STORE_DWORDX2 $vgpr1_vgpr2, $vgpr3_vgpr4, 0, 0, implicit $exec
    $vgpr4 = GLOBAL_LOAD_DWORD $vgpr4_vgpr5, 0, 0, implicit $exec
    GLOBAL_STORE_DWORD $vgpr0_vgpr1, $vgpr2, 0, 0, implicit $exec
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("Drain");
  MachineBasicBlock &BB0 = *MF.getBlockNumbered(0);

  auto &ET = visitAll(BB0);

  EXPECT_EQ(ET.getPendingEvents(AMDGPU::STORE_CNT),
            HWEvents(HWEvents::VMEM_WRITE_ACCESS));
  EXPECT_EQ(ET.getPendingEvents(AMDGPU::LOAD_CNT),
            HWEvents(HWEvents::VMEM_READ_ACCESS));
  EXPECT_FALSE(ET.getPendingEvents(AMDGPU::DS_CNT));

  // Partial drain: only the two youngest stores remain.
  ET.drain(AMDGPU::STORE_CNT, 2);
  {
    auto StoreCnt = TrackerRecordsChecker(ET, AMDGPU::STORE_CNT);
    EXPECT_EQ(StoreCnt.getCount(), 2u);
    EXPECT_FALSE(StoreCnt.empty());
    EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(),
              AMDGPU::GLOBAL_STORE_DWORDX2);
    EXPECT_TRUE(StoreCnt.next());
    EXPECT_EQ(StoreCnt.cur().getMI()->getOpcode(), AMDGPU::GLOBAL_STORE_DWORD);
    EXPECT_FALSE(StoreCnt.next());
  }

  // Draining a counter does not affect other counters.
  EXPECT_EQ(ET.count(AMDGPU::LOAD_CNT), 1u);
  EXPECT_EQ(ET.getTimeline(AMDGPU::LOAD_CNT).size(), 1u);

  // Draining above the current count is a no-op.
  ET.drain(AMDGPU::STORE_CNT, 5);
  EXPECT_EQ(ET.count(AMDGPU::STORE_CNT), 2u);
  EXPECT_EQ(ET.getTimeline(AMDGPU::STORE_CNT).size(), 2u);

  // Full drain.
  ET.drain(AMDGPU::STORE_CNT);
  EXPECT_TRUE(TrackerRecordsChecker(ET, AMDGPU::STORE_CNT).unused());
  EXPECT_FALSE(ET.getPendingEvents(AMDGPU::STORE_CNT));
}

} // namespace
