//===- TraceBufferTest.cpp - TraceBufferCore unit tests ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OMPT/TraceBuffer.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

using namespace llvm::omp::target::ompt;

namespace {

struct TestRec {
  uint64_t Seq;
};
using TestManager = TraceBufferManager<TestRec>;

// Thread-safe: producer threads and the helper thread may call these
// concurrently.
class FakeTool : public BufferProvider, public RecordConsumer {
public:
  explicit FakeTool(size_t SlotsPerBuffer)
      : BufferBytes(SlotsPerBuffer * TestManager::getStride()) {}

  bool ReturnNull = false;
  bool ReturnTooSmall = false;

  BufferTy allocateBuffer(int64_t DeviceId) override {
    std::lock_guard<std::mutex> Lock(Mutex);
    ++AllocateCallCount;
    if (ReturnNull)
      return {nullptr, 0};
    if (ReturnTooSmall)
      return {reinterpret_cast<void *>(alignof(std::max_align_t)), 1};
    Buffers.emplace_back(BufferBytes);
    HandedOut.emplace_back(DeviceId, Buffers.back().data());
    return {Buffers.back().data(), BufferBytes};
  }

  void releaseBuffer(int64_t DeviceId, void *Buffer) override {
    std::lock_guard<std::mutex> Lock(Mutex);
    Released.emplace_back(DeviceId, Buffer);
  }

  void consumeRecords(int64_t DeviceId, void *, void *First,
                      size_t Bytes) override {
    std::vector<uint64_t> Seqs;
    for (auto *R = static_cast<TestRec *>(First);
         reinterpret_cast<char *>(R) < static_cast<char *>(First) + Bytes;
         R = TestManager::next(R))
      Seqs.push_back(R->Seq);
    std::lock_guard<std::mutex> Lock(Mutex);
    auto &DeviceSeqs = ConsumedByDevice[DeviceId];
    DeviceSeqs.insert(DeviceSeqs.end(), Seqs.begin(), Seqs.end());
    ConsumeCalls.push_back(std::move(Seqs));
  }

  int allocateCalls() const {
    std::lock_guard<std::mutex> Lock(Mutex);
    return AllocateCallCount;
  }
  size_t releasedCount() const {
    std::lock_guard<std::mutex> Lock(Mutex);
    return Released.size();
  }
  std::vector<std::vector<uint64_t>> consumeCalls() const {
    std::lock_guard<std::mutex> Lock(Mutex);
    return ConsumeCalls;
  }
  std::vector<uint64_t> allConsumed() const {
    std::lock_guard<std::mutex> Lock(Mutex);
    std::vector<uint64_t> All;
    for (auto &Call : ConsumeCalls)
      All.insert(All.end(), Call.begin(), Call.end());
    return All;
  }
  void *lastBufferStart() const {
    std::lock_guard<std::mutex> Lock(Mutex);
    return const_cast<std::byte *>(Buffers.back().data());
  }
  void *bufferStart(size_t I) const {
    std::lock_guard<std::mutex> Lock(Mutex);
    return HandedOut.at(I).second;
  }
  // Records of DeviceId in consumption order.
  std::vector<uint64_t> consumedFor(int64_t DeviceId) const {
    std::lock_guard<std::mutex> Lock(Mutex);
    auto It = ConsumedByDevice.find(DeviceId);
    return It == ConsumedByDevice.end() ? std::vector<uint64_t>{} : It->second;
  }
  std::vector<void *> released() const {
    std::lock_guard<std::mutex> Lock(Mutex);
    std::vector<void *> Ptrs;
    for (auto &[DeviceId, Ptr] : Released)
      Ptrs.push_back(Ptr);
    return Ptrs;
  }
  size_t releasedCountFor(int64_t DeviceId) const {
    std::lock_guard<std::mutex> Lock(Mutex);
    return std::count_if(Released.begin(), Released.end(),
                         [DeviceId](auto &R) { return R.first == DeviceId; });
  }
  // Every release is of a buffer handed out for the same device, at most once.
  testing::AssertionResult releasesAreValid() const {
    std::lock_guard<std::mutex> Lock(Mutex);
    std::map<void *, int64_t> Owner;
    for (auto &[DeviceId, Ptr] : HandedOut)
      Owner.emplace(Ptr, DeviceId);
    std::map<void *, int> Count;
    for (auto &[DeviceId, Ptr] : Released) {
      auto It = Owner.find(Ptr);
      if (It == Owner.end())
        return testing::AssertionFailure()
               << "released a buffer never handed out: " << Ptr;
      if (It->second != DeviceId)
        return testing::AssertionFailure()
               << "buffer " << Ptr << " of device " << It->second
               << " released for device " << DeviceId;
      if (++Count[Ptr] > 1)
        return testing::AssertionFailure()
               << "buffer " << Ptr << " released twice";
    }
    return testing::AssertionSuccess();
  }

private:
  size_t BufferBytes;
  mutable std::mutex Mutex;
  std::deque<std::vector<std::byte>> Buffers;
  std::vector<std::pair<int64_t, void *>> HandedOut;
  std::vector<std::vector<uint64_t>> ConsumeCalls;
  std::map<int64_t, std::vector<uint64_t>> ConsumedByDevice;
  std::vector<std::pair<int64_t, void *>> Released;
  int AllocateCallCount = 0;
};

std::vector<uint64_t> sorted(std::vector<uint64_t> V) {
  std::sort(V.begin(), V.end());
  return V;
}

} // namespace

TEST(TraceBufferTest, PublishedRecordsReachConsumerInOrderAfterFlush) {
  FakeTool Tool(4);
  TestManager Mgr(Tool, Tool);
  for (uint64_t I = 0; I < 3; ++I) {
    TestRec *R = Mgr.alloc(0);
    ASSERT_NE(R, nullptr);
    R->Seq = I;
    Mgr.publish(R);
  }
  EXPECT_TRUE(Mgr.getCore().flush(0));
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0, 1, 2}));
  EXPECT_EQ(Tool.releasedCount(), 0u);
}

TEST(TraceBufferTest, UnpublishedRecordsAreHeldBackThenConsumedExactlyOnce) {
  FakeTool Tool(4);
  TestManager Mgr(Tool, Tool);
  TestRec *R0 = Mgr.alloc(0);
  ASSERT_NE(R0, nullptr);
  R0->Seq = 0;
  Mgr.publish(R0);
  TestRec *R1 = Mgr.alloc(0);
  ASSERT_NE(R1, nullptr);
  TestRec *R2 = Mgr.alloc(0);
  ASSERT_NE(R2, nullptr);
  R2->Seq = 2;
  Mgr.publish(R2);

  Mgr.getCore().flush(0);
  EXPECT_EQ(sorted(Tool.allConsumed()), (std::vector<uint64_t>{0, 2}));

  R1->Seq = 1;
  Mgr.publish(R1);
  Mgr.getCore().flush(0);
  EXPECT_EQ(sorted(Tool.allConsumed()), (std::vector<uint64_t>{0, 1, 2}));
}

TEST(TraceBufferTest, UnpublishedSlotSplitsARunIntoTwoDeliveries) {
  FakeTool Tool(4);
  TestManager Mgr(Tool, Tool);
  TestRec *R0 = Mgr.alloc(0);
  R0->Seq = 0;
  Mgr.publish(R0);
  TestRec *R1 = Mgr.alloc(0);
  (void)R1; // left unpublished
  TestRec *R2 = Mgr.alloc(0);
  R2->Seq = 2;
  Mgr.publish(R2);

  Mgr.getCore().flush(0);
  auto Calls = Tool.consumeCalls();
  ASSERT_EQ(Calls.size(), 2u);
  EXPECT_EQ(Calls[0], (std::vector<uint64_t>{0}));
  EXPECT_EQ(Calls[1], (std::vector<uint64_t>{2}));
}

TEST(TraceBufferTest, FullBufferAutoQueuedWhenFlushOnBufferFullEnabled) {
  FakeTool Tool(2);
  TestManager Mgr(Tool, Tool);
  TestRec *R0 = Mgr.alloc(0);
  R0->Seq = 0;
  Mgr.publish(R0);
  TestRec *R1 = Mgr.alloc(0);
  R1->Seq = 1;
  Mgr.publish(R1);
  TestRec *R2 = Mgr.alloc(0); // buffer 1 is now full; triggers auto-queue.
  ASSERT_NE(R2, nullptr);
  (void)R2; // left unpublished: irrelevant, lives in a fresh buffer.

  EXPECT_TRUE(Mgr.getCore().processPending());
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0, 1}));
}

TEST(TraceBufferTest, FullBufferWaitsForFlushWhenFlushOnBufferFullDisabled) {
  FakeTool Tool(2);
  TraceBufferConfig Cfg;
  Cfg.FlushOnBufferFull = false;
  Cfg.FlushOnShutdown = false; // Only the explicit flush below delivers.
  TestManager Mgr(Tool, Tool, Cfg);
  TestRec *R0 = Mgr.alloc(0);
  R0->Seq = 0;
  Mgr.publish(R0);
  TestRec *R1 = Mgr.alloc(0);
  R1->Seq = 1;
  Mgr.publish(R1);
  TestRec *R2 = Mgr.alloc(0); // buffer 1 is now full, but not queued.
  ASSERT_NE(R2, nullptr);
  R2->Seq = 2;
  Mgr.publish(R2);

  EXPECT_FALSE(Mgr.getCore().processPending());
  EXPECT_TRUE(Tool.consumeCalls().empty());

  EXPECT_TRUE(Mgr.getCore().flush(0));
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0, 1, 2}));
  // Buffer 1 is retired; buffer 2 (holding R2) is not full and stays.
  EXPECT_EQ(Tool.released(), (std::vector<void *>{Tool.bufferStart(0)}));
  EXPECT_TRUE(Tool.releasesAreValid());
}

TEST(TraceBufferTest, ShutdownDeliversWhenFlushOnBufferFullDisabled) {
  FakeTool Tool(2);
  TraceBufferConfig Cfg;
  Cfg.FlushOnBufferFull = false;
  TestManager Mgr(Tool, Tool, Cfg);
  // Device 0 fills buffers 1 and 2 and starts 3; device 1 fills one buffer
  // and starts a second.
  for (uint64_t Seq = 0; Seq < 5; ++Seq) {
    TestRec *R = Mgr.alloc(0);
    ASSERT_NE(R, nullptr);
    R->Seq = Seq;
    Mgr.publish(R);
  }
  for (uint64_t Seq = 10; Seq < 13; ++Seq) {
    TestRec *R = Mgr.alloc(1);
    ASSERT_NE(R, nullptr);
    R->Seq = Seq;
    Mgr.publish(R);
  }
  EXPECT_TRUE(Tool.consumeCalls().empty());

  Mgr.getCore().flushAndStopHelper(); // No helper running: flushes inline.
  EXPECT_EQ(Tool.consumedFor(0), (std::vector<uint64_t>{0, 1, 2, 3, 4}));
  EXPECT_EQ(Tool.consumedFor(1), (std::vector<uint64_t>{10, 11, 12}));
  EXPECT_EQ(Tool.releasedCountFor(0), 2u);
  EXPECT_EQ(Tool.releasedCountFor(1), 1u);
  EXPECT_TRUE(Tool.releasesAreValid());
}

TEST(TraceBufferTest,
     UnflushedFullBuffersAreKeptNotDroppedWhenFlushOnBufferFullDisabled) {
  FakeTool Tool(2);
  TraceBufferConfig Cfg;
  Cfg.FlushOnBufferFull = false;
  TestManager Mgr(Tool, Tool, Cfg);
  // Fills buffers 1-4 and starts buffer 5; nobody flushes.
  for (uint64_t Seq = 0; Seq < 9; ++Seq) {
    TestRec *R = Mgr.alloc(0);
    ASSERT_NE(R, nullptr);
    R->Seq = Seq;
    Mgr.publish(R);
  }
  // A full buffer is set aside, and the producer gets a new one.
  EXPECT_EQ(Tool.allocateCalls(), 5);
  EXPECT_FALSE(Mgr.getCore().processPending());
  EXPECT_TRUE(Tool.consumeCalls().empty());
  EXPECT_EQ(Tool.releasedCount(), 0u);

  EXPECT_TRUE(Mgr.getCore().flushAll());
  EXPECT_EQ(Tool.allConsumed(),
            (std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_EQ(Tool.released(),
            (std::vector<void *>{Tool.bufferStart(0), Tool.bufferStart(1),
                                 Tool.bufferStart(2), Tool.bufferStart(3)}));
  EXPECT_TRUE(Tool.releasesAreValid());
}

TEST(TraceBufferTest, FullyConsumedFullBufferIsReleasedExactlyOnce) {
  FakeTool Tool(2);
  TestManager Mgr(Tool, Tool);
  TestRec *R0 = Mgr.alloc(0);
  R0->Seq = 0;
  Mgr.publish(R0);
  TestRec *R1 = Mgr.alloc(0);
  R1->Seq = 1;
  Mgr.publish(R1);
  TestRec *R2 = Mgr.alloc(0); // fills and auto-queues buffer 1.
  ASSERT_NE(R2, nullptr);
  (void)R2; // left unpublished

  Mgr.getCore().processPending();
  EXPECT_EQ(Tool.releasedCount(), 1u);

  // Buffer 2 (holding R2) is not full: must never be released.
  Mgr.getCore().flush(0);
  EXPECT_EQ(Tool.releasedCount(), 1u);
}

TEST(TraceBufferTest, FullBufferWithUnpublishedSlotRetiresOnceCompleted) {
  FakeTool Tool(2);
  TestManager Mgr(Tool, Tool);
  TestRec *R0 = Mgr.alloc(0);
  R0->Seq = 0;
  Mgr.publish(R0);
  TestRec *R1 = Mgr.alloc(0); // left unpublished
  TestRec *R2 = Mgr.alloc(0); // fills and auto-queues buffer 1.
  ASSERT_NE(R2, nullptr);
  (void)R2; // left unpublished

  EXPECT_TRUE(Mgr.getCore().processPending());
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0}));
  EXPECT_EQ(Tool.releasedCount(), 0u);

  // R1 is still unpublished: nothing new for a pass to find.
  EXPECT_FALSE(Mgr.getCore().processPending());

  R1->Seq = 1;
  Mgr.publish(R1);
  EXPECT_TRUE(Mgr.getCore().processPending());
  EXPECT_EQ(sorted(Tool.allConsumed()), (std::vector<uint64_t>{0, 1}));
  EXPECT_EQ(Tool.releasedCount(), 1u);

  // Buffer retired, queue now empty: nothing new to process.
  EXPECT_FALSE(Mgr.getCore().processPending());
}

TEST(TraceBufferTest, NullProviderBufferYieldsNullAllocAndRegistersNothing) {
  FakeTool Tool(4);
  Tool.ReturnNull = true;
  TestManager Mgr(Tool, Tool);
  EXPECT_EQ(Mgr.alloc(0), nullptr);
  EXPECT_EQ(Tool.allocateCalls(), 1);

  Tool.ReturnNull = false;
  TestRec *R = Mgr.alloc(0);
  ASSERT_NE(R, nullptr);
  EXPECT_EQ(static_cast<void *>(R), Tool.lastBufferStart());
  EXPECT_TRUE(Tool.consumeCalls().empty());
}

TEST(TraceBufferTest,
     TooSmallProviderBufferYieldsNullAllocAndRegistersNothing) {
  FakeTool Tool(4);
  Tool.ReturnTooSmall = true;
  TestManager Mgr(Tool, Tool);
  EXPECT_EQ(Mgr.alloc(0), nullptr);
  EXPECT_EQ(Tool.allocateCalls(), 1);

  Tool.ReturnTooSmall = false;
  TestRec *R = Mgr.alloc(0);
  ASSERT_NE(R, nullptr);
  EXPECT_EQ(static_cast<void *>(R), Tool.lastBufferStart());
}

TEST(TraceBufferTest, FullBufferAutoQueuedWhenReplacementDeclined) {
  FakeTool Tool(1);
  TestManager Mgr(Tool, Tool);
  TestRec *R0 = Mgr.alloc(0);
  ASSERT_NE(R0, nullptr);
  R0->Seq = 0;
  Mgr.publish(R0);

  Tool.ReturnNull = true;
  // Buffer 1 is now full and its replacement is declined.
  EXPECT_EQ(Mgr.alloc(0), nullptr);

  EXPECT_TRUE(Mgr.getCore().processPending());
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0}));
  EXPECT_EQ(Tool.releasedCount(), 1u);
  EXPECT_TRUE(Tool.releasesAreValid());

  Tool.ReturnNull = false;
  TestRec *R1 = Mgr.alloc(0);
  ASSERT_NE(R1, nullptr);
  EXPECT_EQ(static_cast<void *>(R1), Tool.lastBufferStart());
  R1->Seq = 1;
  Mgr.publish(R1);
  EXPECT_TRUE(Mgr.getCore().flushAll());
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0, 1}));
}

TEST(TraceBufferTest, OutOfRangeDeviceIdYieldsNullWithoutCallingProvider) {
  FakeTool Tool(4);
  TraceBufferConfig Cfg;
  Cfg.MaxDevices = 4;
  TestManager Mgr(Tool, Tool, Cfg);
  EXPECT_EQ(Mgr.alloc(-1), nullptr);
  EXPECT_EQ(Mgr.alloc(4), nullptr);
  EXPECT_EQ(Tool.allocateCalls(), 0);
}

TEST(TraceBufferTest, FlushOutOfRangeDeviceIdReturnsFalse) {
  FakeTool Tool(4);
  TraceBufferConfig Cfg;
  Cfg.MaxDevices = 4;
  TestManager Mgr(Tool, Tool, Cfg);
  EXPECT_FALSE(Mgr.getCore().flush(-1));
  EXPECT_FALSE(Mgr.getCore().flush(4));
}

TEST(TraceBufferTest, FlushOneDeviceLeavesOtherDeviceUntouched) {
  FakeTool Tool(4);
  TestManager Mgr(Tool, Tool);
  TestRec *R0 = Mgr.alloc(0);
  R0->Seq = 0;
  Mgr.publish(R0);
  TestRec *R1 = Mgr.alloc(1);
  R1->Seq = 100;
  Mgr.publish(R1);

  Mgr.getCore().flush(0);
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0}));

  Mgr.getCore().flushAll();
  EXPECT_EQ(sorted(Tool.allConsumed()), (std::vector<uint64_t>{0, 100}));
}

namespace {
class ReentrantFakeTool : public FakeTool {
public:
  explicit ReentrantFakeTool(size_t SlotsPerBuffer)
      : FakeTool(SlotsPerBuffer) {}

  TraceBufferCore *Core = nullptr;
  bool ReentrantFlushResult = true;

  void consumeRecords(int64_t DeviceId, void *Buffer, void *First,
                      size_t Bytes) override {
    FakeTool::consumeRecords(DeviceId, Buffer, First, Bytes);
    if (Core)
      ReentrantFlushResult = Core->flush(DeviceId);
  }
};
} // namespace

TEST(TraceBufferTest, FlushFromInsideConsumeRecordsReturnsFalse) {
  ReentrantFakeTool Tool(4);
  TestManager Mgr(Tool, Tool);
  Tool.Core = &Mgr.getCore();
  TestRec *R0 = Mgr.alloc(0);
  R0->Seq = 0;
  Mgr.publish(R0);

  EXPECT_TRUE(Mgr.getCore().flush(0));
  EXPECT_FALSE(Tool.ReentrantFlushResult);
}

namespace {
// A's consumer flushes a different core (B) inline, then re-enters A's own
// flush from the same callback.
class CrossCoreFakeTool : public FakeTool {
public:
  explicit CrossCoreFakeTool(size_t SlotsPerBuffer)
      : FakeTool(SlotsPerBuffer) {}

  TraceBufferCore *CoreA = nullptr;
  TraceBufferCore *CoreB = nullptr;
  bool ReentrantAFlushResult = true;

  void consumeRecords(int64_t DeviceId, void *Buffer, void *First,
                      size_t Bytes) override {
    FakeTool::consumeRecords(DeviceId, Buffer, First, Bytes);
    if (CoreB)
      CoreB->flush(DeviceId);
    if (CoreA)
      ReentrantAFlushResult = CoreA->flush(DeviceId);
  }
};
} // namespace

TEST(TraceBufferTest, NestedFlushOfAnotherCoreDoesNotBreakReentrancyGuard) {
  CrossCoreFakeTool ToolA(4);
  FakeTool ToolB(4);
  TestManager MgrA(ToolA, ToolA);
  TestManager MgrB(ToolB, ToolB);
  ToolA.CoreA = &MgrA.getCore();
  ToolA.CoreB = &MgrB.getCore();

  TestRec *RB = MgrB.alloc(0);
  ASSERT_NE(RB, nullptr);
  RB->Seq = 100;
  MgrB.publish(RB);

  TestRec *RA = MgrA.alloc(0);
  ASSERT_NE(RA, nullptr);
  RA->Seq = 1;
  MgrA.publish(RA);

  EXPECT_TRUE(MgrA.getCore().flush(0));
  EXPECT_FALSE(ToolA.ReentrantAFlushResult);
  EXPECT_EQ(ToolB.allConsumed(), (std::vector<uint64_t>{100}));

  // A still works normally after the nested callback.
  TestRec *RA2 = MgrA.alloc(0);
  ASSERT_NE(RA2, nullptr);
  RA2->Seq = 2;
  MgrA.publish(RA2);
  EXPECT_TRUE(MgrA.getCore().flush(0));
  EXPECT_EQ(sorted(ToolA.allConsumed()), (std::vector<uint64_t>{1, 2}));
}

TEST(TraceBufferTest, DestructionDropsOutstandingBuffersSilently) {
  FakeTool Tool(4);
  {
    TestManager Mgr(Tool, Tool);
    TestRec *R0 = Mgr.alloc(0);
    ASSERT_NE(R0, nullptr);
    (void)R0; // left unpublished (Init): dropped, no callbacks.
  }
  EXPECT_TRUE(Tool.consumeCalls().empty());
  EXPECT_EQ(Tool.releasedCount(), 0u);
}

TEST(TraceBufferTest, TwoCoresOnOneThreadDoNotShareBuffers) {
  FakeTool ToolA(4), ToolB(4);
  TestManager MgrA(ToolA, ToolA);
  TestManager MgrB(ToolB, ToolB);

  TestRec *RA = MgrA.alloc(0);
  ASSERT_NE(RA, nullptr);
  RA->Seq = 1;
  MgrA.publish(RA);
  TestRec *RB = MgrB.alloc(0);
  ASSERT_NE(RB, nullptr);
  RB->Seq = 2;
  MgrB.publish(RB);

  MgrA.getCore().flush(0);
  MgrB.getCore().flush(0);
  EXPECT_EQ(ToolA.allConsumed(), (std::vector<uint64_t>{1}));
  EXPECT_EQ(ToolB.allConsumed(), (std::vector<uint64_t>{2}));
}

TEST(TraceBufferTest, NewCoreAfterDestroyedOneStartsClean) {
  FakeTool Tool1(4);
  {
    TestManager Mgr1(Tool1, Tool1);
    TestRec *R = Mgr1.alloc(0);
    ASSERT_NE(R, nullptr);
    (void)R; // left unpublished: destroyed without flushing.
  }

  FakeTool Tool2(4);
  TestManager Mgr2(Tool2, Tool2);
  TestRec *R2 = Mgr2.alloc(0);
  ASSERT_NE(R2, nullptr);
  R2->Seq = 2;
  Mgr2.publish(R2);
  Mgr2.getCore().flush(0);
  EXPECT_EQ(Tool2.allConsumed(), (std::vector<uint64_t>{2}));
}

TEST(TraceBufferTest, TypedWrapperStrideNextAndSlotLayout) {
  FakeTool Tool(4);
  TestManager Mgr(Tool, Tool);
  TestRec *R0 = Mgr.alloc(0);
  ASSERT_NE(R0, nullptr);
  R0->Seq = 42;

  TestRec *R1 = TestManager::next(R0);
  EXPECT_EQ(reinterpret_cast<char *>(R1) - reinterpret_cast<char *>(R0),
            static_cast<ptrdiff_t>(TestManager::getStride()));
  EXPECT_EQ(static_cast<void *>(R0), Tool.lastBufferStart());
}

#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
namespace {
class MisalignedTool : public BufferProvider, public RecordConsumer {
public:
  explicit MisalignedTool(size_t Bytes) : Storage(Bytes + 1) {}

  BufferTy allocateBuffer(int64_t) override {
    // Deliberately misaligned.
    return {Storage.data() + 1, Storage.size() - 1};
  }
  void releaseBuffer(int64_t, void *) override {}
  void consumeRecords(int64_t, void *, void *, size_t) override {}

private:
  std::vector<std::byte> Storage;
};
} // namespace

TEST(TraceBufferTest, MisalignedProviderBufferAsserts) {
  MisalignedTool Tool(4 * TestManager::getStride());
  TestManager Mgr(Tool, Tool);
  EXPECT_DEATH({ Mgr.alloc(0); }, "not malloc-aligned");
}

TEST(TraceBufferTest, DestroyingWithRunningHelperAsserts) {
  EXPECT_DEATH(
      {
        FakeTool Tool(4);
        TestManager Mgr(Tool, Tool);
        Mgr.getCore().startHelper();
        // Mgr destroyed here with the helper still running.
      },
      "Helper still running");
}

TEST(TraceBufferTest, DestroyingWithUnconsumedReadyRecordAsserts) {
  EXPECT_DEATH(
      {
        FakeTool Tool(4);
        TestManager Mgr(Tool, Tool); // FlushOnShutdown defaults true.
        TestRec *R = Mgr.alloc(0);
        R->Seq = 0;
        Mgr.publish(R);
        // Mgr destroyed here with a Ready, never-consumed record.
      },
      "Unconsumed record");
}

namespace {
class SelfProcessingTool : public FakeTool {
public:
  using FakeTool::FakeTool;
  TraceBufferCore *Core = nullptr;

  void consumeRecords(int64_t DeviceId, void *Buffer, void *First,
                      size_t Bytes) override {
    FakeTool::consumeRecords(DeviceId, Buffer, First, Bytes);
    Core->processPending();
  }
};

class BlockingTool : public FakeTool {
public:
  using FakeTool::FakeTool;
  std::mutex M;
  std::condition_variable Cv;
  bool Inside = false;

  void consumeRecords(int64_t DeviceId, void *Buffer, void *First,
                      size_t Bytes) override {
    FakeTool::consumeRecords(DeviceId, Buffer, First, Bytes);
    std::unique_lock<std::mutex> Lock(M);
    Inside = true;
    Cv.notify_all();
    Cv.wait(Lock, [] { return false; });
  }
};
} // namespace

TEST(TraceBufferTest, ProcessPendingFromOwnConsumerAsserts) {
  EXPECT_DEATH(
      {
        SelfProcessingTool Tool(4);
        TestManager Mgr(Tool, Tool);
        Tool.Core = &Mgr.getCore();
        TestRec *R = Mgr.alloc(0);
        R->Seq = 0;
        Mgr.publish(R);
        Mgr.getCore().flush(0);
      },
      "Not re-entered from its own callback");
}

TEST(TraceBufferTest, LifecycleCallDuringFlushAsserts) {
  EXPECT_DEATH(
      {
        BlockingTool Tool(4);
        TestManager Mgr(Tool, Tool);
        TestRec *R = Mgr.alloc(0);
        R->Seq = 0;
        Mgr.publish(R);
        std::thread Flusher([&] { Mgr.getCore().flush(0); });
        {
          std::unique_lock<std::mutex> Lock(Tool.M);
          Tool.Cv.wait(Lock, [&] { return Tool.Inside; });
        }
        Mgr.getCore().startHelper();
      },
      "Flushes and lifecycle calls do not overlap");
}
#endif

TEST(TraceBufferTest, UnflushedRecordsAtDestructionDoNotAssertWhenAllowed) {
  {
    FakeTool Tool(4);
    TraceBufferConfig Cfg;
    Cfg.FlushOnShutdown = false;
    TestManager Mgr(Tool, Tool, Cfg);
    TestRec *R = Mgr.alloc(0);
    ASSERT_NE(R, nullptr);
    R->Seq = 0;
    Mgr.publish(R); // Ready, but FlushOnShutdown is off: no assert.
  }
  {
    FakeTool Tool(4);
    TestManager Mgr(Tool, Tool); // FlushOnShutdown defaults true.
    TestRec *R = Mgr.alloc(0);
    ASSERT_NE(R, nullptr);
    (void)R; // Init (unpublished): dropped, no assert either way.
  }
}

TEST(TraceBufferTest, ConcurrentProducersFlushAllDeliversEveryRecordOnce) {
  constexpr int NumThreads = 4;
  constexpr int RecordsPerThread = 200;
  FakeTool Tool(8);
  TestManager Mgr(Tool, Tool);
  Mgr.getCore().setEnabled(true);
  Mgr.getCore().startHelper();

  std::vector<std::thread> Producers;
  for (int T = 0; T < NumThreads; ++T) {
    Producers.emplace_back([&Mgr, T] {
      for (int I = 0; I < RecordsPerThread; ++I) {
        TestRec *R = Mgr.alloc(0);
        EXPECT_NE(R, nullptr);
        if (!R)
          return;
        R->Seq = static_cast<uint64_t>(T) * 1000000 + static_cast<uint64_t>(I);
        Mgr.publish(R);
      }
    });
  }
  for (auto &Th : Producers)
    Th.join();

  EXPECT_TRUE(Mgr.getCore().flushAll());
  Mgr.getCore().stopHelper();

  auto Consumed = sorted(Tool.allConsumed());
  ASSERT_EQ(Consumed.size(),
            static_cast<size_t>(NumThreads * RecordsPerThread));
  EXPECT_EQ(std::unique(Consumed.begin(), Consumed.end()), Consumed.end());

  std::map<uint64_t, std::vector<uint64_t>> ByThread;
  for (auto &Call : Tool.consumeCalls())
    for (uint64_t S : Call)
      ByThread[S / 1000000].push_back(S % 1000000);
  for (auto &Entry : ByThread)
    EXPECT_TRUE(std::is_sorted(Entry.second.begin(), Entry.second.end()));
}

TEST(TraceBufferTest, RecordPublishedOnAnotherThreadIsConsumed) {
  FakeTool Tool(4);
  TestManager Mgr(Tool, Tool);
  Mgr.getCore().setEnabled(true);
  Mgr.getCore().startHelper();

  TestRec *R = Mgr.alloc(0);
  ASSERT_NE(R, nullptr);
  R->Seq = 7;

  std::thread Publisher([&Mgr, R] { Mgr.publish(R); });
  Publisher.join();

  EXPECT_TRUE(Mgr.getCore().flush(0));
  Mgr.getCore().stopHelper();

  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{7}));
}

TEST(TraceBufferTest, DisabledHelperDoesNotProcessUntilEnabledOrFlushed) {
  FakeTool Tool(2);
  TestManager Mgr(Tool, Tool);
  Mgr.getCore().startHelper(); // Enabled defaults to false.

  TestRec *R0 = Mgr.alloc(0);
  R0->Seq = 0;
  Mgr.publish(R0);
  TestRec *R1 = Mgr.alloc(0);
  R1->Seq = 1;
  Mgr.publish(R1);
  TestRec *R2 = Mgr.alloc(0); // fills and auto-queues buffer 1.
  ASSERT_NE(R2, nullptr);
  (void)R2; // left unpublished

  // Disabled: the helper cannot act, regardless of scheduling.
  EXPECT_TRUE(Tool.consumeCalls().empty());

  // No flush() call: this must rely on setEnabled's own wakeup.
  Mgr.getCore().setEnabled(true);

  auto Deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (Tool.allConsumed().size() < 2) {
    ASSERT_LT(std::chrono::steady_clock::now(), Deadline)
        << "timed out waiting for consumption after enabling";
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  Mgr.getCore().stopHelper();
  EXPECT_EQ(sorted(Tool.allConsumed()), (std::vector<uint64_t>{0, 1}));
}

TEST(TraceBufferTest, EnabledHelperReScansAndRetiresAfterLatePublish) {
  FakeTool Tool(2);
  TraceBufferConfig Cfg;
  Cfg.HelperPollInterval = std::chrono::milliseconds(1);
  TestManager Mgr(Tool, Tool, Cfg);
  Mgr.getCore().setEnabled(true);
  Mgr.getCore().startHelper();

  TestRec *R0 = Mgr.alloc(0);
  R0->Seq = 0;
  Mgr.publish(R0);
  TestRec *R1 = Mgr.alloc(0); // left unpublished
  TestRec *R2 = Mgr.alloc(0); // forces buffer 1 full and auto-queues it.
  ASSERT_NE(R2, nullptr);
  (void)R2; // left unpublished: lives in a fresh, never-queued buffer.

  auto Deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (Tool.allConsumed().empty()) {
    ASSERT_LT(std::chrono::steady_clock::now(), Deadline)
        << "timed out waiting for the first partial pass";
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  R1->Seq = 1;
  Mgr.publish(R1);

  while (Tool.releasedCount() == 0) {
    ASSERT_LT(std::chrono::steady_clock::now(), Deadline)
        << "timed out waiting for retirement after the late publish";
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  Mgr.getCore().stopHelper();
  EXPECT_EQ(sorted(Tool.allConsumed()), (std::vector<uint64_t>{0, 1}));
  EXPECT_EQ(Tool.releasedCount(), 1u);
}

TEST(TraceBufferTest, FlushAndStopHelperWithFlushOnShutdownConsumesEverything) {
  FakeTool Tool(4);
  TestManager Mgr(Tool, Tool);
  Mgr.getCore().setEnabled(true);
  Mgr.getCore().startHelper();

  TestRec *R = Mgr.alloc(0);
  ASSERT_NE(R, nullptr);
  R->Seq = 5;
  Mgr.publish(R);

  Mgr.getCore().flushAndStopHelper();
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{5}));
}

TEST(TraceBufferTest,
     FlushAndStopHelperWithoutFlushOnShutdownConsumesNothingNew) {
  FakeTool Tool(4);
  TraceBufferConfig Cfg;
  Cfg.FlushOnShutdown = false;
  TestManager Mgr(Tool, Tool, Cfg);
  Mgr.getCore().setEnabled(true);
  Mgr.getCore().startHelper();

  TestRec *R = Mgr.alloc(0);
  ASSERT_NE(R, nullptr);
  R->Seq = 5;
  Mgr.publish(R);

  Mgr.getCore().flushAndStopHelper();
  EXPECT_TRUE(Tool.allConsumed().empty());
}

TEST(TraceBufferTest, StopHelperRunsAFinalPassEvenWhileDisabled) {
  FakeTool Tool(2);
  TraceBufferConfig Cfg;
  Cfg.FlushOnShutdown = false; // R2's buffer is deliberately left unconsumed.
  TestManager Mgr(Tool, Tool, Cfg);
  Mgr.getCore().startHelper(); // Enabled defaults to false.

  TestRec *R0 = Mgr.alloc(0);
  R0->Seq = 0;
  Mgr.publish(R0);
  TestRec *R1 = Mgr.alloc(0);
  R1->Seq = 1;
  Mgr.publish(R1);
  TestRec *R2 = Mgr.alloc(0); // fills and auto-queues buffer 1.
  ASSERT_NE(R2, nullptr);
  (void)R2; // left unpublished

  // Disabled: only stop or a flush request can wake the helper, so nothing
  // races between the auto-queue above and stopHelper() below.
  Mgr.getCore().stopHelper();
  EXPECT_EQ(sorted(Tool.allConsumed()), (std::vector<uint64_t>{0, 1}));
  EXPECT_EQ(Tool.releasedCount(), 1u);
}

TEST(TraceBufferTest, HelperLifecycleStartStopStartAndStopWithoutStart) {
  FakeTool Tool(4);
  TestManager Mgr(Tool, Tool);

  Mgr.getCore().stopHelper(); // No-op: never started.

  Mgr.getCore().startHelper();
  Mgr.getCore().setEnabled(true);
  TestRec *R0 = Mgr.alloc(0);
  ASSERT_NE(R0, nullptr);
  R0->Seq = 1;
  Mgr.publish(R0);
  EXPECT_TRUE(Mgr.getCore().flush(0));
  Mgr.getCore().stopHelper();

  Mgr.getCore().startHelper();
  TestRec *R1 = Mgr.alloc(0);
  ASSERT_NE(R1, nullptr);
  R1->Seq = 2;
  Mgr.publish(R1);
  EXPECT_TRUE(Mgr.getCore().flush(0));
  Mgr.getCore().stopHelper();

  EXPECT_EQ(sorted(Tool.allConsumed()), (std::vector<uint64_t>{1, 2}));
}

TEST(TraceBufferTest, FlushWithNoBuffersConsumesNothing) {
  FakeTool Tool(4);
  TestManager Mgr(Tool, Tool);
  EXPECT_TRUE(Mgr.getCore().flush(0));
  EXPECT_TRUE(Mgr.getCore().flushAll());
  EXPECT_TRUE(Tool.consumeCalls().empty());
  EXPECT_EQ(Tool.releasedCount(), 0u);
}

TEST(TraceBufferTest, PartialBufferQueuedByFlushRetiresOnceFilled) {
  FakeTool Tool(2);
  TestManager Mgr(Tool, Tool);
  TestRec *R0 = Mgr.alloc(0);
  R0->Seq = 0;
  Mgr.publish(R0);
  // Queues buffer 0 while it is partial; its entry stays queued after the pass.
  EXPECT_TRUE(Mgr.getCore().flush(0));
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0}));

  TestRec *R1 = Mgr.alloc(0);
  R1->Seq = 1;
  Mgr.publish(R1);
  TestRec *R2 = Mgr.alloc(0); // fills buffer 0, which is already queued.
  ASSERT_NE(R2, nullptr);
  (void)R2; // left unpublished

  EXPECT_TRUE(Mgr.getCore().processPending());
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0, 1}));
  EXPECT_EQ(Tool.released(), (std::vector<void *>{Tool.bufferStart(0)}));
  EXPECT_FALSE(Mgr.getCore().processPending());
}

namespace {
// Blocks the BlockOnCall-th allocateBuffer call until open() is called.
class GatedFakeTool : public FakeTool {
public:
  GatedFakeTool(size_t SlotsPerBuffer, int BlockOnCall)
      : FakeTool(SlotsPerBuffer), BlockOnCall(BlockOnCall) {}

  BufferTy allocateBuffer(int64_t DeviceId) override {
    {
      std::unique_lock<std::mutex> Lock(GateMutex);
      if (++Calls == BlockOnCall) {
        Blocked = true;
        GateCv.notify_all();
        GateCv.wait(Lock, [this] { return Open; });
      }
    }
    return FakeTool::allocateBuffer(DeviceId);
  }

  void waitUntilBlocked() {
    std::unique_lock<std::mutex> Lock(GateMutex);
    GateCv.wait(Lock, [this] { return Blocked; });
  }
  void open() {
    {
      std::lock_guard<std::mutex> Lock(GateMutex);
      Open = true;
    }
    GateCv.notify_all();
  }

private:
  const int BlockOnCall;
  std::mutex GateMutex;
  std::condition_variable GateCv;
  int Calls = 0;
  bool Blocked = false;
  bool Open = false;
};
} // namespace

TEST(TraceBufferTest, FullBufferRetiredBeforeItsProducerQueuesItIsNotRequeued) {
  GatedFakeTool Tool(2, /*BlockOnCall=*/2);
  TestManager Mgr(Tool, Tool);
  TestRec *R2 = nullptr;
  std::thread Producer([&] {
    for (uint64_t I = 0; I < 2; ++I) {
      TestRec *R = Mgr.alloc(0);
      ASSERT_NE(R, nullptr);
      R->Seq = I;
      Mgr.publish(R);
    }
    // Marks buffer 0 full, then blocks in the provider before queuing it.
    R2 = Mgr.alloc(0);
  });

  Tool.waitUntilBlocked();
  EXPECT_TRUE(Mgr.getCore().flush(0));
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0, 1}));
  EXPECT_EQ(Tool.released(), (std::vector<void *>{Tool.bufferStart(0)}));

  Tool.open();
  Producer.join();
  ASSERT_NE(R2, nullptr);
  R2->Seq = 2;
  Mgr.publish(R2);
  EXPECT_TRUE(Mgr.getCore().flush(0));
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0, 1, 2}));
  EXPECT_EQ(Tool.released(), (std::vector<void *>{Tool.bufferStart(0)}));
}

TEST(TraceBufferTest, FlushSkipsRetiredBufferIdBetweenHeldBackAndCurrent) {
  FakeTool Tool(2);
  TestManager Mgr(Tool, Tool);
  TestRec *R0 = Mgr.alloc(0);
  R0->Seq = 0;
  Mgr.publish(R0);
  TestRec *R1 = Mgr.alloc(0); // held back: buffer 0 cannot retire yet.
  TestRec *R2 = Mgr.alloc(0); // buffer 0 full; buffer 1.
  R2->Seq = 2;
  Mgr.publish(R2);
  TestRec *R3 = Mgr.alloc(0);
  R3->Seq = 3;
  Mgr.publish(R3);
  TestRec *R4 = Mgr.alloc(0); // buffer 1 full; buffer 2 is current.
  ASSERT_NE(R4, nullptr);

  // Retires buffer 1, leaving buffer ids 0 and 2 registered.
  EXPECT_TRUE(Mgr.getCore().processPending());
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0, 2, 3}));
  EXPECT_EQ(Tool.released(), (std::vector<void *>{Tool.bufferStart(1)}));

  R4->Seq = 4;
  Mgr.publish(R4);
  EXPECT_TRUE(Mgr.getCore().flush(0));
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0, 2, 3, 4}));
  EXPECT_EQ(Tool.released(), (std::vector<void *>{Tool.bufferStart(1)}));

  R1->Seq = 1;
  Mgr.publish(R1);
  EXPECT_TRUE(Mgr.getCore().flush(0));
  EXPECT_EQ(Tool.allConsumed(), (std::vector<uint64_t>{0, 2, 3, 4, 1}));
  EXPECT_EQ(Tool.released(),
            (std::vector<void *>{Tool.bufferStart(1), Tool.bufferStart(0)}));
  EXPECT_TRUE(Tool.releasesAreValid());
}

TEST(TraceBufferTest, InterleavedDevicesFlushedPerDeviceDeliverEachInOrder) {
  constexpr int64_t NumDevices = 3;
  constexpr uint64_t Base = 1000;
  FakeTool Tool(2);
  TraceBufferConfig Cfg;
  Cfg.MaxDevices = NumDevices;
  TestManager Mgr(Tool, Tool, Cfg);
  auto Produce = [&](uint64_t From, uint64_t To) {
    for (uint64_t I = From; I < To; ++I)
      for (int64_t D = 0; D < NumDevices; ++D) {
        TestRec *R = Mgr.alloc(D);
        ASSERT_NE(R, nullptr);
        R->Seq = static_cast<uint64_t>(D) * Base + I;
        Mgr.publish(R);
      }
  };
  auto Expected = [&](int64_t D, uint64_t N) {
    std::vector<uint64_t> Seqs;
    for (uint64_t I = 0; I < N; ++I)
      Seqs.push_back(static_cast<uint64_t>(D) * Base + I);
    return Seqs;
  };
  auto Contains = [](const std::vector<uint64_t> &V, uint64_t S) {
    return std::find(V.begin(), V.end(), S) != V.end();
  };

  // Record 4 of each device sits alone in its current, never queued buffer.
  Produce(0, 5);
  EXPECT_TRUE(Mgr.getCore().flush(1));
  EXPECT_EQ(Tool.consumedFor(1), Expected(1, 5));
  EXPECT_FALSE(Contains(Tool.consumedFor(0), 0 * Base + 4));
  EXPECT_FALSE(Contains(Tool.consumedFor(2), 2 * Base + 4));
  EXPECT_TRUE(Mgr.getCore().flush(0));
  EXPECT_EQ(Tool.consumedFor(0), Expected(0, 5));
  EXPECT_FALSE(Contains(Tool.consumedFor(2), 2 * Base + 4));
  EXPECT_TRUE(Mgr.getCore().flush(2));
  EXPECT_EQ(Tool.consumedFor(2), Expected(2, 5));

  Produce(5, 8);
  EXPECT_TRUE(Mgr.getCore().flush(2));
  EXPECT_TRUE(Mgr.getCore().flush(0));
  EXPECT_TRUE(Mgr.getCore().flush(1));
  for (int64_t D = 0; D < NumDevices; ++D) {
    EXPECT_EQ(Tool.consumedFor(D), Expected(D, 8));
    // Four buffers per device; all but the current one are full.
    EXPECT_EQ(Tool.releasedCountFor(D), 3u);
  }
  EXPECT_EQ(Tool.allocateCalls(), 4 * NumDevices);
  EXPECT_TRUE(Tool.releasesAreValid());
}

TEST(TraceBufferTest, FlushesRacingProducersAndHelperDeliverAndReleaseOnce) {
  constexpr int NumProducers = 4;
  constexpr int64_t NumDevices = 3;
  constexpr uint64_t RecordsPerDevice = 500;
  constexpr size_t SlotsPerBuffer = 2;
  constexpr uint64_t Base = 1000000;
  FakeTool Tool(SlotsPerBuffer);
  TraceBufferConfig Cfg;
  Cfg.MaxDevices = NumDevices;
  Cfg.HelperPollInterval = std::chrono::milliseconds(1);
  TestManager Mgr(Tool, Tool, Cfg);
  Mgr.getCore().setEnabled(true);
  Mgr.getCore().startHelper();

  std::atomic<int> ProducersLeft{NumProducers};
  std::vector<std::thread> Producers;
  for (int P = 0; P < NumProducers; ++P)
    Producers.emplace_back([&, P] {
      for (uint64_t I = 0; I < RecordsPerDevice; ++I)
        for (int64_t D = 0; D < NumDevices; ++D) {
          TestRec *R = Mgr.alloc(D);
          EXPECT_NE(R, nullptr);
          if (!R)
            continue;
          R->Seq = static_cast<uint64_t>(P * NumDevices + D) * Base + I;
          Mgr.publish(R);
        }
      --ProducersLeft;
    });

  std::vector<std::thread> Flushers;
  for (int64_t D = 0; D <= NumDevices; ++D)
    Flushers.emplace_back([&, D] {
      do {
        if (D == NumDevices)
          EXPECT_TRUE(Mgr.getCore().flushAll());
        else
          EXPECT_TRUE(Mgr.getCore().flush(D));
      } while (ProducersLeft.load() > 0);
    });

  for (auto &Th : Producers)
    Th.join();
  for (auto &Th : Flushers)
    Th.join();
  Mgr.getCore().flushAndStopHelper();

  // Order is only guaranteed per producer and device, not across devices.
  for (int64_t D = 0; D < NumDevices; ++D) {
    std::map<int, std::vector<uint64_t>> ByProducer;
    for (uint64_t S : Tool.consumedFor(D)) {
      uint64_t Key = S / Base;
      ASSERT_EQ(static_cast<int64_t>(Key % NumDevices), D)
          << "record " << S << " delivered for the wrong device";
      ByProducer[static_cast<int>(Key / NumDevices)].push_back(S % Base);
    }
    EXPECT_EQ(ByProducer.size(), static_cast<size_t>(NumProducers));
    std::vector<uint64_t> All(RecordsPerDevice);
    for (uint64_t I = 0; I < RecordsPerDevice; ++I)
      All[I] = I;
    for (int P = 0; P < NumProducers; ++P)
      EXPECT_EQ(ByProducer[P], All) << "producer " << P << ", device " << D;
  }

  // Each producer's current buffer per device is never full, so never
  // released; every other buffer is full with all slots published, so the
  // final flush retires it.
  constexpr uint64_t BuffersPerPair =
      (RecordsPerDevice + SlotsPerBuffer - 1) / SlotsPerBuffer;
  constexpr int NumPairs = NumProducers * NumDevices;
  EXPECT_EQ(Tool.allocateCalls(), static_cast<int>(NumPairs * BuffersPerPair));
  EXPECT_EQ(Tool.releasedCount(),
            static_cast<size_t>(NumPairs * (BuffersPerPair - 1)));
  EXPECT_TRUE(Tool.releasesAreValid());
}
