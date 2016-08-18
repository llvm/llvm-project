//===-- StreamTest.cpp - Tests for Stream ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the unit tests for Stream code.
///
//===----------------------------------------------------------------------===//

#include <cstring>

#include "streamexecutor/Executor.h"
#include "streamexecutor/Kernel.h"
#include "streamexecutor/KernelSpec.h"
#include "streamexecutor/PlatformInterfaces.h"
#include "streamexecutor/Stream.h"

#include "gtest/gtest.h"

namespace {

namespace se = ::streamexecutor;

/// Mock PlatformExecutor that performs asynchronous memcpy operations by
/// ignoring the stream argument and calling std::memcpy on device memory
/// handles.
class MockPlatformExecutor : public se::PlatformExecutor {
public:
  ~MockPlatformExecutor() override {}

  std::string getName() const override { return "MockPlatformExecutor"; }

  se::Expected<std::unique_ptr<se::PlatformStreamHandle>>
  createStream() override {
    return nullptr;
  }

  se::Error memcpyD2H(se::PlatformStreamHandle *,
                      const se::GlobalDeviceMemoryBase &DeviceSrc,
                      void *HostDst, size_t ByteCount) override {
    std::memcpy(HostDst, DeviceSrc.getHandle(), ByteCount);
    return se::Error::success();
  }

  se::Error memcpyH2D(se::PlatformStreamHandle *, const void *HostSrc,
                      se::GlobalDeviceMemoryBase *DeviceDst,
                      size_t ByteCount) override {
    std::memcpy(const_cast<void *>(DeviceDst->getHandle()), HostSrc, ByteCount);
    return se::Error::success();
  }

  se::Error memcpyD2D(se::PlatformStreamHandle *,
                      const se::GlobalDeviceMemoryBase &DeviceSrc,
                      se::GlobalDeviceMemoryBase *DeviceDst,
                      size_t ByteCount) override {
    std::memcpy(const_cast<void *>(DeviceDst->getHandle()),
                DeviceSrc.getHandle(), ByteCount);
    return se::Error::success();
  }
};

/// Test fixture to hold objects used by tests.
class StreamTest : public ::testing::Test {
public:
  StreamTest()
      : DeviceA(se::GlobalDeviceMemory<int>::makeFromElementCount(HostA, 10)),
        DeviceB(se::GlobalDeviceMemory<int>::makeFromElementCount(HostB, 10)),
        Stream(llvm::make_unique<se::PlatformStreamHandle>(&PExecutor)) {}

protected:
  // Device memory is backed by host arrays.
  int HostA[10];
  se::GlobalDeviceMemory<int> DeviceA;
  int HostB[10];
  se::GlobalDeviceMemory<int> DeviceB;

  // Host memory to be used as actual host memory.
  int Host[10];

  MockPlatformExecutor PExecutor;
  se::Stream Stream;
};

TEST_F(StreamTest, MemcpyCorrectSize) {
  Stream.thenMemcpyH2D(llvm::ArrayRef<int>(Host), &DeviceA);
  EXPECT_TRUE(Stream.isOK());

  Stream.thenMemcpyD2H(DeviceA, llvm::MutableArrayRef<int>(Host));
  EXPECT_TRUE(Stream.isOK());

  Stream.thenMemcpyD2D(DeviceA, &DeviceB);
  EXPECT_TRUE(Stream.isOK());
}

TEST_F(StreamTest, MemcpyH2DTooManyElements) {
  Stream.thenMemcpyH2D(llvm::ArrayRef<int>(Host), &DeviceA, 20);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, MemcpyD2HTooManyElements) {
  Stream.thenMemcpyD2H(DeviceA, llvm::MutableArrayRef<int>(Host), 20);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, MemcpyD2DTooManyElements) {
  Stream.thenMemcpyD2D(DeviceA, &DeviceB, 20);
  EXPECT_FALSE(Stream.isOK());
}

} // namespace
