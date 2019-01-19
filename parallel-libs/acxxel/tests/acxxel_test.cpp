//===--- acxxel_test.cpp - Tests for the Acxxel API -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "acxxel.h"
#include "config.h"
#include "gtest/gtest.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace {

template <typename T, size_t N> constexpr size_t arraySize(T (&)[N]) {
  return N;
}

using PlatformGetter = acxxel::Expected<acxxel::Platform *> (*)();
class AcxxelTest : public ::testing::TestWithParam<PlatformGetter> {};

TEST_P(AcxxelTest, GetDeviceCount) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  int DeviceCount = Platform->getDeviceCount().getValue();
  EXPECT_GE(DeviceCount, 0);
}

// Tests all the methods of a DeviceMemorySpan that was created from the asSpan
// method of a DeviceMemory object.
//
// The length is the number of elements in the span. The ElementByteSize is the
// number of bytes per element in the span.
//
// It is assumed that the input span has 10 or more elements.
template <typename SpanType>
void testFullDeviceMemorySpan(SpanType &&Span, ptrdiff_t Length,
                              ptrdiff_t ElementByteSize) {
  EXPECT_GE(Length, 10);
  EXPECT_GT(ElementByteSize, 0);

  // Full span
  EXPECT_EQ(Length, Span.length());
  EXPECT_EQ(Length, Span.size());
  EXPECT_EQ(Length * ElementByteSize, Span.byte_size());
  EXPECT_EQ(0, Span.offset());
  EXPECT_EQ(0, Span.byte_offset());
  EXPECT_FALSE(Span.empty());

  // Sub-span with first method.
  auto First2 = Span.first(2);
  EXPECT_EQ(2, First2.length());
  EXPECT_EQ(2, First2.size());
  EXPECT_EQ(2 * ElementByteSize, First2.byte_size());
  EXPECT_EQ(0, First2.offset());
  EXPECT_EQ(0, First2.byte_offset());
  EXPECT_FALSE(First2.empty());

  auto First0 = Span.first(0);
  EXPECT_EQ(0, First0.length());
  EXPECT_EQ(0, First0.size());
  EXPECT_EQ(0, First0.byte_size());
  EXPECT_EQ(0, First0.offset());
  EXPECT_EQ(0, First0.byte_offset());
  EXPECT_TRUE(First0.empty());

  // Sub-span with last method.
  auto Last2 = Span.last(2);
  EXPECT_EQ(2, Last2.length());
  EXPECT_EQ(2, Last2.size());
  EXPECT_EQ(2 * ElementByteSize, Last2.byte_size());
  EXPECT_EQ(Length - 2, Last2.offset());
  EXPECT_EQ((Length - 2) * ElementByteSize, Last2.byte_offset());
  EXPECT_FALSE(Last2.empty());

  auto Last0 = Span.last(0);
  EXPECT_EQ(0, Last0.length());
  EXPECT_EQ(0, Last0.size());
  EXPECT_EQ(0, Last0.byte_size());
  EXPECT_EQ(Length, Last0.offset());
  EXPECT_EQ(Length * ElementByteSize, Last0.byte_offset());
  EXPECT_TRUE(Last0.empty());

  // Sub-span with subspan method.
  auto Middle2 = Span.subspan(4, 2);
  EXPECT_EQ(2, Middle2.length());
  EXPECT_EQ(2, Middle2.size());
  EXPECT_EQ(2 * ElementByteSize, Middle2.byte_size());
  EXPECT_EQ(4, Middle2.offset());
  EXPECT_EQ(4 * ElementByteSize, Middle2.byte_offset());
  EXPECT_FALSE(Middle2.empty());

  auto Middle0 = Span.subspan(4, 0);
  EXPECT_EQ(0, Middle0.length());
  EXPECT_EQ(0, Middle0.size());
  EXPECT_EQ(0, Middle0.byte_size());
  EXPECT_EQ(4, Middle0.offset());
  EXPECT_EQ(4 * ElementByteSize, Middle0.byte_offset());
  EXPECT_TRUE(Middle0.empty());

  auto Subspan2AtStart = Span.subspan(0, 2);
  EXPECT_EQ(2, Subspan2AtStart.length());
  EXPECT_EQ(2, Subspan2AtStart.size());
  EXPECT_EQ(2 * ElementByteSize, Subspan2AtStart.byte_size());
  EXPECT_EQ(0, Subspan2AtStart.offset());
  EXPECT_EQ(0, Subspan2AtStart.byte_offset());
  EXPECT_FALSE(Subspan2AtStart.empty());

  auto Subspan2AtEnd = Span.subspan(Length - 2, 2);
  EXPECT_EQ(2, Subspan2AtEnd.length());
  EXPECT_EQ(2, Subspan2AtEnd.size());
  EXPECT_EQ(2 * ElementByteSize, Subspan2AtEnd.byte_size());
  EXPECT_EQ(Length - 2, Subspan2AtEnd.offset());
  EXPECT_EQ((Length - 2) * ElementByteSize, Subspan2AtEnd.byte_offset());
  EXPECT_FALSE(Subspan2AtEnd.empty());

  auto Subspan0AtStart = Span.subspan(0, 0);
  EXPECT_EQ(0, Subspan0AtStart.length());
  EXPECT_EQ(0, Subspan0AtStart.size());
  EXPECT_EQ(0, Subspan0AtStart.byte_size());
  EXPECT_EQ(0, Subspan0AtStart.offset());
  EXPECT_EQ(0, Subspan0AtStart.byte_offset());
  EXPECT_TRUE(Subspan0AtStart.empty());

  auto Subspan0AtEnd = Span.subspan(Length, 0);
  EXPECT_EQ(0, Subspan0AtEnd.length());
  EXPECT_EQ(0, Subspan0AtEnd.size());
  EXPECT_EQ(0, Subspan0AtEnd.byte_size());
  EXPECT_EQ(Length, Subspan0AtEnd.offset());
  EXPECT_EQ(Length * ElementByteSize, Subspan0AtEnd.byte_offset());
  EXPECT_TRUE(Subspan0AtEnd.empty());
}

TEST_P(AcxxelTest, DeviceMemory) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  acxxel::Expected<acxxel::DeviceMemory<int>> MaybeMemory =
      Platform->mallocD<int>(10);
  EXPECT_FALSE(MaybeMemory.isError());

  // ref
  acxxel::DeviceMemory<int> &MemoryRef = MaybeMemory.getValue();
  EXPECT_EQ(10, MemoryRef.length());
  EXPECT_EQ(10, MemoryRef.size());
  EXPECT_EQ(10 * sizeof(int), static_cast<size_t>(MemoryRef.byte_size()));
  EXPECT_FALSE(MemoryRef.empty());

  // mutable span
  acxxel::DeviceMemorySpan<int> MutableSpan = MemoryRef.asSpan();
  testFullDeviceMemorySpan(MutableSpan, 10, sizeof(int));

  // const ref
  const acxxel::DeviceMemory<int> &ConstMemoryRef = MaybeMemory.getValue();
  EXPECT_EQ(10, ConstMemoryRef.length());
  EXPECT_EQ(10, ConstMemoryRef.size());
  EXPECT_EQ(10 * sizeof(int), static_cast<size_t>(ConstMemoryRef.byte_size()));
  EXPECT_FALSE(ConstMemoryRef.empty());

  // immutable span
  acxxel::DeviceMemorySpan<const int> ImmutableSpan = ConstMemoryRef.asSpan();
  testFullDeviceMemorySpan(ImmutableSpan, 10, sizeof(int));
}

TEST_P(AcxxelTest, CopyHostAndDevice) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  acxxel::Stream Stream = Platform->createStream().takeValue();
  int A[] = {0, 1, 2};
  std::array<int, arraySize(A)> B;
  acxxel::DeviceMemory<int> X =
      Platform->mallocD<int>(arraySize(A)).takeValue();
  Stream.syncCopyHToD(A, X);
  Stream.syncCopyDToH(X, B);
  for (size_t I = 0; I < arraySize(A); ++I)
    EXPECT_EQ(A[I], B[I]);
  EXPECT_FALSE(Stream.takeStatus().isError());
}

TEST_P(AcxxelTest, CopyDToD) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  acxxel::Stream Stream = Platform->createStream().takeValue();
  int A[] = {0, 1, 2};
  std::array<int, arraySize(A)> B;
  acxxel::DeviceMemory<int> X =
      Platform->mallocD<int>(arraySize(A)).takeValue();
  acxxel::DeviceMemory<int> Y =
      Platform->mallocD<int>(arraySize(A)).takeValue();
  Stream.syncCopyHToD(A, X);
  Stream.syncCopyDToD(X, Y);
  Stream.syncCopyDToH(Y, B);
  for (size_t I = 0; I < arraySize(A); ++I)
    EXPECT_EQ(A[I], B[I]);
  EXPECT_FALSE(Stream.takeStatus().isError());
}

TEST_P(AcxxelTest, AsyncCopyHostAndDevice) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  int A[] = {0, 1, 2};
  std::array<int, arraySize(A)> B;
  acxxel::DeviceMemory<int> X =
      Platform->mallocD<int>(arraySize(A)).takeValue();
  acxxel::Stream Stream = Platform->createStream().takeValue();
  acxxel::AsyncHostMemory<int> AsyncA =
      Platform->registerHostMem(A).takeValue();
  acxxel::AsyncHostMemory<int> AsyncB =
      Platform->registerHostMem(B).takeValue();
  EXPECT_FALSE(Stream.asyncCopyHToD(AsyncA, X).takeStatus().isError());
  EXPECT_FALSE(Stream.asyncCopyDToH(X, AsyncB).takeStatus().isError());
  EXPECT_FALSE(Stream.sync().isError());
  for (size_t I = 0; I < arraySize(A); ++I)
    EXPECT_EQ(A[I], B[I]);
}

TEST_P(AcxxelTest, AsyncMemsetD) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  constexpr size_t ArrayLength = 10;
  std::array<uint32_t, ArrayLength> Host;
  acxxel::DeviceMemory<uint32_t> X =
      Platform->mallocD<uint32_t>(ArrayLength).takeValue();
  acxxel::Stream Stream = Platform->createStream().takeValue();
  acxxel::AsyncHostMemory<uint32_t> AsyncHost =
      Platform->registerHostMem(Host).takeValue();
  EXPECT_FALSE(Stream.asyncMemsetD(X, 0x12).takeStatus().isError());
  EXPECT_FALSE(Stream.asyncCopyDToH(X, AsyncHost).takeStatus().isError());
  EXPECT_FALSE(Stream.sync().isError());
  for (size_t I = 0; I < ArrayLength; ++I)
    EXPECT_EQ(0x12121212u, Host[I]);
}

TEST_P(AcxxelTest, RegisterHostMem) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  auto Data = std::unique_ptr<int[]>(new int[3]);
  acxxel::Expected<acxxel::AsyncHostMemory<const int>> MaybeAsyncHostMemory =
      Platform->registerHostMem<int>({Data.get(), 3});
  EXPECT_FALSE(MaybeAsyncHostMemory.isError())
      << MaybeAsyncHostMemory.getError().getMessage();
  acxxel::AsyncHostMemory<const int> AsyncHostMemory =
      MaybeAsyncHostMemory.takeValue();
  EXPECT_EQ(Data.get(), AsyncHostMemory.data());
  EXPECT_EQ(3, AsyncHostMemory.size());
}

struct RefCounter {
  static int Count;

  RefCounter() { ++Count; }
  ~RefCounter() { --Count; }
  RefCounter(const RefCounter &) = delete;
  RefCounter &operator=(const RefCounter &) = delete;
};

int RefCounter::Count;

TEST_P(AcxxelTest, OwnedAsyncHost) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  RefCounter::Count = 0;
  {
    acxxel::OwnedAsyncHostMemory<RefCounter> A =
        Platform->newAsyncHostMem<RefCounter>(3).takeValue();
    EXPECT_EQ(3, RefCounter::Count);
  }
  EXPECT_EQ(0, RefCounter::Count);
}

TEST_P(AcxxelTest, OwnedAsyncCopyHostAndDevice) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  size_t Length = 3;
  acxxel::OwnedAsyncHostMemory<int> A =
      Platform->newAsyncHostMem<int>(Length).takeValue();
  for (size_t I = 0; I < Length; ++I)
    A[I] = I;
  acxxel::OwnedAsyncHostMemory<int> B =
      Platform->newAsyncHostMem<int>(Length).takeValue();
  acxxel::DeviceMemory<int> X = Platform->mallocD<int>(Length).takeValue();
  acxxel::Stream Stream = Platform->createStream().takeValue();
  EXPECT_FALSE(Stream.asyncCopyHToD(A, X).takeStatus().isError());
  EXPECT_FALSE(Stream.asyncCopyDToH(X, B).takeStatus().isError());
  EXPECT_FALSE(Stream.sync().isError());
  for (size_t I = 0; I < Length; ++I)
    EXPECT_EQ(A[I], B[I]);
}

TEST_P(AcxxelTest, AsyncCopyDToD) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  int A[] = {0, 1, 2};
  std::array<int, arraySize(A)> B;
  acxxel::DeviceMemory<int> X =
      Platform->mallocD<int>(arraySize(A)).takeValue();
  acxxel::DeviceMemory<int> Y =
      Platform->mallocD<int>(arraySize(A)).takeValue();
  acxxel::Stream Stream = Platform->createStream().takeValue();
  acxxel::AsyncHostMemory<int> AsyncA =
      Platform->registerHostMem(A).takeValue();
  acxxel::AsyncHostMemory<int> AsyncB =
      Platform->registerHostMem(B).takeValue();
  EXPECT_FALSE(Stream.asyncCopyHToD(AsyncA, X).takeStatus().isError());
  EXPECT_FALSE(Stream.asyncCopyDToD(X, Y).takeStatus().isError());
  EXPECT_FALSE(Stream.asyncCopyDToH(Y, AsyncB).takeStatus().isError());
  EXPECT_FALSE(Stream.sync().isError());
  for (size_t I = 0; I < arraySize(A); ++I)
    EXPECT_EQ(A[I], B[I]);
}

TEST_P(AcxxelTest, Stream) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  acxxel::Stream Stream = Platform->createStream().takeValue();
  EXPECT_FALSE(Stream.sync().isError());
}

TEST_P(AcxxelTest, Event) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  acxxel::Event Event = Platform->createEvent().takeValue();
  EXPECT_TRUE(Event.isDone());
  EXPECT_FALSE(Event.sync().isError());
}

TEST_P(AcxxelTest, RecordEventsInAStream) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  acxxel::Stream Stream = Platform->createStream().takeValue();
  acxxel::Event Start = Platform->createEvent().takeValue();
  acxxel::Event End = Platform->createEvent().takeValue();
  EXPECT_FALSE(Stream.enqueueEvent(Start).takeStatus().isError());
  EXPECT_FALSE(Start.sync().isError());
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(Stream.enqueueEvent(End).takeStatus().isError());
  EXPECT_FALSE(End.sync().isError());
  EXPECT_GT(End.getSecondsSince(Start).takeValue(), 0);
}

TEST_P(AcxxelTest, StreamCallback) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  int Value = 0;
  acxxel::Stream Stream = Platform->createStream().takeValue();
  EXPECT_FALSE(
      Stream
          .addCallback([&Value](acxxel::Stream &, const acxxel::Status &) {
            Value = 42;
          })
          .takeStatus()
          .isError());
  EXPECT_FALSE(Stream.sync().isError());
  EXPECT_EQ(42, Value);
}

TEST_P(AcxxelTest, WaitForEventsInAStream) {
  acxxel::Platform *Platform = GetParam()().takeValue();
  acxxel::Stream Stream0 = Platform->createStream().takeValue();
  acxxel::Stream Stream1 = Platform->createStream().takeValue();
  acxxel::Event Event0 = Platform->createEvent().takeValue();
  acxxel::Event Event1 = Platform->createEvent().takeValue();

  // Thread loops on Stream0 until someone sets the GoFlag, then set the
  // MarkerFlag.

  std::mutex Mutex;
  std::condition_variable ConditionVar;
  bool GoFlag = false;
  bool MarkerFlag = false;

  EXPECT_FALSE(Stream0
                   .addCallback([&Mutex, &ConditionVar, &GoFlag, &MarkerFlag](
                       acxxel::Stream &, const acxxel::Status &) {
                     std::unique_lock<std::mutex> Lock(Mutex);
                     ConditionVar.wait(Lock,
                                       [&GoFlag] { return GoFlag == true; });
                     MarkerFlag = true;
                   })
                   .takeStatus()
                   .isError());

  // Event0 can only occur after GoFlag and MarkerFlag are set.
  EXPECT_FALSE(Stream0.enqueueEvent(Event0).takeStatus().isError());

  // Use waitOnEvent to make a callback on Stream1 wait for an event on Stream0.
  EXPECT_FALSE(Stream1.waitOnEvent(Event0).isError());
  EXPECT_FALSE(Stream1.enqueueEvent(Event1).takeStatus().isError());
  EXPECT_FALSE(Stream1
                   .addCallback([&Mutex, &MarkerFlag](acxxel::Stream &,
                                                      const acxxel::Status &) {
                     std::unique_lock<std::mutex> Lock(Mutex);
                     // This makes sure that this callback runs after the
                     // callback on Stream0.
                     EXPECT_TRUE(MarkerFlag);
                   })
                   .takeStatus()
                   .isError());

  // Allow the callback on Stream0 to set MarkerFlag and finish.
  {
    std::unique_lock<std::mutex> Lock(Mutex);
    GoFlag = true;
  }
  ConditionVar.notify_one();

  // Make sure the events have finished and that Event1 did not happen before
  // Event0.
  EXPECT_FALSE(Event0.sync().isError());
  EXPECT_FALSE(Event1.sync().isError());
  EXPECT_FALSE(Stream1.sync().isError());
}

#if defined(ACXXEL_ENABLE_CUDA) || defined(ACXXEL_ENABLE_OPENCL)
INSTANTIATE_TEST_CASE_P(BothPlatformTest, AcxxelTest,
                        ::testing::Values(
#ifdef ACXXEL_ENABLE_CUDA
                            acxxel::getCUDAPlatform
#ifdef ACXXEL_ENABLE_OPENCL
                            ,
#endif
#endif
#ifdef ACXXEL_ENABLE_OPENCL
                            acxxel::getOpenCLPlatform
#endif
                            ));
#endif

} // namespace
