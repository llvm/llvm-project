//===-- DeviceTest.cpp - Tests for Device ---------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the unit tests for Device code.
///
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <cstring>

#include "streamexecutor/Device.h"
#include "streamexecutor/PlatformInterfaces.h"

#include "gtest/gtest.h"

namespace {

namespace se = ::streamexecutor;

class MockPlatformDevice : public se::PlatformDevice {
public:
  ~MockPlatformDevice() override {}

  std::string getName() const override { return "MockPlatformDevice"; }

  se::Expected<std::unique_ptr<se::PlatformStreamHandle>>
  createStream() override {
    return se::make_error("not implemented");
  }

  se::Expected<se::GlobalDeviceMemoryBase>
  allocateDeviceMemory(size_t ByteCount) override {
    return se::GlobalDeviceMemoryBase(std::malloc(ByteCount));
  }

  se::Error freeDeviceMemory(se::GlobalDeviceMemoryBase Memory) override {
    std::free(const_cast<void *>(Memory.getHandle()));
    return se::Error::success();
  }

  se::Expected<void *> allocateHostMemory(size_t ByteCount) override {
    return std::malloc(ByteCount);
  }

  se::Error freeHostMemory(void *Memory) override {
    std::free(Memory);
    return se::Error::success();
  }

  se::Error registerHostMemory(void *, size_t) override {
    return se::Error::success();
  }

  se::Error unregisterHostMemory(void *) override {
    return se::Error::success();
  }

  se::Error synchronousCopyD2H(const se::GlobalDeviceMemoryBase &DeviceSrc,
                               size_t SrcByteOffset, void *HostDst,
                               size_t DstByteOffset,
                               size_t ByteCount) override {
    std::memcpy(static_cast<char *>(HostDst) + DstByteOffset,
                static_cast<const char *>(DeviceSrc.getHandle()) +
                    SrcByteOffset,
                ByteCount);
    return se::Error::success();
  }

  se::Error synchronousCopyH2D(const void *HostSrc, size_t SrcByteOffset,
                               se::GlobalDeviceMemoryBase DeviceDst,
                               size_t DstByteOffset,
                               size_t ByteCount) override {
    std::memcpy(static_cast<char *>(const_cast<void *>(DeviceDst.getHandle())) +
                    DstByteOffset,
                static_cast<const char *>(HostSrc) + SrcByteOffset, ByteCount);
    return se::Error::success();
  }

  se::Error synchronousCopyD2D(se::GlobalDeviceMemoryBase DeviceDst,
                               size_t DstByteOffset,
                               const se::GlobalDeviceMemoryBase &DeviceSrc,
                               size_t SrcByteOffset,
                               size_t ByteCount) override {
    std::memcpy(static_cast<char *>(const_cast<void *>(DeviceDst.getHandle())) +
                    DstByteOffset,
                static_cast<const char *>(DeviceSrc.getHandle()) +
                    SrcByteOffset,
                ByteCount);
    return se::Error::success();
  }
};

/// Test fixture to hold objects used by tests.
class DeviceTest : public ::testing::Test {
public:
  DeviceTest()
      : HostA5{0, 1, 2, 3, 4}, HostB5{5, 6, 7, 8, 9},
        HostA7{10, 11, 12, 13, 14, 15, 16}, HostB7{17, 18, 19, 20, 21, 22, 23},
        DeviceA5(se::GlobalDeviceMemory<int>::makeFromElementCount(HostA5, 5)),
        DeviceB5(se::GlobalDeviceMemory<int>::makeFromElementCount(HostB5, 5)),
        DeviceA7(se::GlobalDeviceMemory<int>::makeFromElementCount(HostA7, 7)),
        DeviceB7(se::GlobalDeviceMemory<int>::makeFromElementCount(HostB7, 7)),
        Host5{24, 25, 26, 27, 28}, Host7{29, 30, 31, 32, 33, 34, 35},
        Device(&PDevice) {}

  // Device memory is backed by host arrays.
  int HostA5[5];
  int HostB5[5];
  int HostA7[7];
  int HostB7[7];
  se::GlobalDeviceMemory<int> DeviceA5;
  se::GlobalDeviceMemory<int> DeviceB5;
  se::GlobalDeviceMemory<int> DeviceA7;
  se::GlobalDeviceMemory<int> DeviceB7;

  // Host memory to be used as actual host memory.
  int Host5[5];
  int Host7[7];

  MockPlatformDevice PDevice;
  se::Device Device;
};

#define EXPECT_NO_ERROR(E) EXPECT_FALSE(static_cast<bool>(E))
#define EXPECT_ERROR(E)                                                        \
  do {                                                                         \
    se::Error E__ = E;                                                         \
    EXPECT_TRUE(static_cast<bool>(E__));                                       \
    consumeError(std::move(E__));                                              \
  } while (false)

using llvm::ArrayRef;
using llvm::MutableArrayRef;

TEST_F(DeviceTest, AllocateAndFreeDeviceMemory) {
  se::Expected<se::GlobalDeviceMemory<int>> MaybeMemory =
      Device.allocateDeviceMemory<int>(10);
  EXPECT_TRUE(static_cast<bool>(MaybeMemory));
  EXPECT_NO_ERROR(Device.freeDeviceMemory(*MaybeMemory));
}

TEST_F(DeviceTest, AllocateAndFreeHostMemory) {
  se::Expected<int *> MaybeMemory = Device.allocateHostMemory<int>(10);
  EXPECT_TRUE(static_cast<bool>(MaybeMemory));
  EXPECT_NO_ERROR(Device.freeHostMemory(*MaybeMemory));
}

TEST_F(DeviceTest, RegisterAndUnregisterHostMemory) {
  std::vector<int> Data(10);
  EXPECT_NO_ERROR(Device.registerHostMemory(Data.data(), 10));
  EXPECT_NO_ERROR(Device.unregisterHostMemory(Data.data()));
}

// D2H tests

TEST_F(DeviceTest, SyncCopyD2HToMutableArrayRefByCount) {
  EXPECT_NO_ERROR(
      Device.synchronousCopyD2H(DeviceA5, MutableArrayRef<int>(Host5), 5));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  EXPECT_NO_ERROR(
      Device.synchronousCopyD2H(DeviceB5, MutableArrayRef<int>(Host5), 2));
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostB5[I], Host5[I]);
  }

  EXPECT_ERROR(
      Device.synchronousCopyD2H(DeviceA7, MutableArrayRef<int>(Host5), 7));

  EXPECT_ERROR(
      Device.synchronousCopyD2H(DeviceA5, MutableArrayRef<int>(Host7), 7));

  EXPECT_ERROR(
      Device.synchronousCopyD2H(DeviceA5, MutableArrayRef<int>(Host5), 7));
}

TEST_F(DeviceTest, SyncCopyD2HToMutableArrayRef) {
  EXPECT_NO_ERROR(
      Device.synchronousCopyD2H(DeviceA5, MutableArrayRef<int>(Host5)));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  EXPECT_ERROR(
      Device.synchronousCopyD2H(DeviceA7, MutableArrayRef<int>(Host5)));

  EXPECT_ERROR(
      Device.synchronousCopyD2H(DeviceA5, MutableArrayRef<int>(Host7)));
}

TEST_F(DeviceTest, SyncCopyD2HToPointer) {
  EXPECT_NO_ERROR(Device.synchronousCopyD2H(DeviceA5, Host5, 5));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyD2H(DeviceA5, Host7, 7));
}

TEST_F(DeviceTest, SyncCopyD2HSliceToMutableArrayRefByCount) {
  EXPECT_NO_ERROR(Device.synchronousCopyD2H(
      DeviceA5.asSlice().drop_front(1), MutableArrayRef<int>(Host5 + 1, 4), 4));
  for (int I = 1; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  EXPECT_NO_ERROR(Device.synchronousCopyD2H(DeviceB5.asSlice().drop_back(1),
                                            MutableArrayRef<int>(Host5), 2));
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostB5[I], Host5[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyD2H(DeviceA7.asSlice(),
                                         MutableArrayRef<int>(Host5), 7));

  EXPECT_ERROR(Device.synchronousCopyD2H(DeviceA5.asSlice(),
                                         MutableArrayRef<int>(Host7), 7));

  EXPECT_ERROR(Device.synchronousCopyD2H(DeviceA5.asSlice(),
                                         MutableArrayRef<int>(Host5), 7));
}

TEST_F(DeviceTest, SyncCopyD2HSliceToMutableArrayRef) {
  EXPECT_NO_ERROR(Device.synchronousCopyD2H(DeviceA7.asSlice().slice(1, 5),
                                            MutableArrayRef<int>(Host5)));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA7[I + 1], Host5[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyD2H(DeviceA7.asSlice().drop_back(1),
                                         MutableArrayRef<int>(Host5)));

  EXPECT_ERROR(Device.synchronousCopyD2H(DeviceA5.asSlice(),
                                         MutableArrayRef<int>(Host7)));
}

TEST_F(DeviceTest, SyncCopyD2HSliceToPointer) {
  EXPECT_NO_ERROR(Device.synchronousCopyD2H(DeviceA5.asSlice().drop_front(1),
                                            Host5 + 1, 4));
  for (int I = 1; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyD2H(DeviceA5.asSlice(), Host7, 7));
}

// H2D tests

TEST_F(DeviceTest, SyncCopyH2DToArrayRefByCount) {
  EXPECT_NO_ERROR(Device.synchronousCopyH2D(ArrayRef<int>(Host5), DeviceA5, 5));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  EXPECT_NO_ERROR(Device.synchronousCopyH2D(ArrayRef<int>(Host5), DeviceB5, 2));
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostB5[I], Host5[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyH2D(ArrayRef<int>(Host7), DeviceA5, 7));

  EXPECT_ERROR(Device.synchronousCopyH2D(ArrayRef<int>(Host5), DeviceA7, 7));

  EXPECT_ERROR(Device.synchronousCopyH2D(ArrayRef<int>(Host5), DeviceA5, 7));
}

TEST_F(DeviceTest, SyncCopyH2DToArrayRef) {
  EXPECT_NO_ERROR(Device.synchronousCopyH2D(ArrayRef<int>(Host5), DeviceA5));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyH2D(ArrayRef<int>(Host5), DeviceA7));

  EXPECT_ERROR(Device.synchronousCopyH2D(ArrayRef<int>(Host7), DeviceA5));
}

TEST_F(DeviceTest, SyncCopyH2DToPointer) {
  EXPECT_NO_ERROR(Device.synchronousCopyH2D(Host5, DeviceA5, 5));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyH2D(Host7, DeviceA5, 7));
}

TEST_F(DeviceTest, SyncCopyH2DSliceToArrayRefByCount) {
  EXPECT_NO_ERROR(Device.synchronousCopyH2D(
      ArrayRef<int>(Host5 + 1, 4), DeviceA5.asSlice().drop_front(1), 4));
  for (int I = 1; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  EXPECT_NO_ERROR(Device.synchronousCopyH2D(
      ArrayRef<int>(Host5), DeviceB5.asSlice().drop_back(1), 2));
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostB5[I], Host5[I]);
  }

  EXPECT_ERROR(
      Device.synchronousCopyH2D(ArrayRef<int>(Host7), DeviceA5.asSlice(), 7));

  EXPECT_ERROR(
      Device.synchronousCopyH2D(ArrayRef<int>(Host5), DeviceA7.asSlice(), 7));

  EXPECT_ERROR(
      Device.synchronousCopyH2D(ArrayRef<int>(Host5), DeviceA5.asSlice(), 7));
}

TEST_F(DeviceTest, SyncCopyH2DSliceToArrayRef) {
  EXPECT_NO_ERROR(
      Device.synchronousCopyH2D(ArrayRef<int>(Host5), DeviceA5.asSlice()));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  EXPECT_ERROR(
      Device.synchronousCopyH2D(ArrayRef<int>(Host5), DeviceA7.asSlice()));

  EXPECT_ERROR(
      Device.synchronousCopyH2D(ArrayRef<int>(Host7), DeviceA5.asSlice()));
}

TEST_F(DeviceTest, SyncCopyH2DSliceToPointer) {
  EXPECT_NO_ERROR(Device.synchronousCopyH2D(Host5, DeviceA5.asSlice(), 5));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyH2D(Host7, DeviceA5.asSlice(), 7));
}

// D2D tests

TEST_F(DeviceTest, SyncCopyD2DByCount) {
  EXPECT_NO_ERROR(Device.synchronousCopyD2D(DeviceA5, DeviceB5, 5));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], HostB5[I]);
  }

  EXPECT_NO_ERROR(Device.synchronousCopyD2D(DeviceA7, DeviceB7, 2));
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostA7[I], HostB7[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA5, DeviceB5, 7));

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA7, DeviceB5, 7));

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA5, DeviceB7, 7));
}

TEST_F(DeviceTest, SyncCopyD2D) {
  EXPECT_NO_ERROR(Device.synchronousCopyD2D(DeviceA5, DeviceB5));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], HostB5[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA7, DeviceB5));

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA5, DeviceB7));
}

TEST_F(DeviceTest, SyncCopySliceD2DByCount) {
  EXPECT_NO_ERROR(
      Device.synchronousCopyD2D(DeviceA5.asSlice().drop_front(1), DeviceB5, 4));
  for (int I = 0; I < 4; ++I) {
    EXPECT_EQ(HostA5[I + 1], HostB5[I]);
  }

  EXPECT_NO_ERROR(
      Device.synchronousCopyD2D(DeviceA7.asSlice().drop_back(1), DeviceB7, 2));
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostA7[I], HostB7[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA5.asSlice(), DeviceB5, 7));

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA7.asSlice(), DeviceB5, 7));

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA5.asSlice(), DeviceB7, 7));
}

TEST_F(DeviceTest, SyncCopySliceD2D) {
  EXPECT_NO_ERROR(
      Device.synchronousCopyD2D(DeviceA7.asSlice().drop_back(2), DeviceB5));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA7[I], HostB5[I]);
  }

  EXPECT_ERROR(
      Device.synchronousCopyD2D(DeviceA7.asSlice().drop_front(1), DeviceB5));

  EXPECT_ERROR(
      Device.synchronousCopyD2D(DeviceA5.asSlice().drop_back(1), DeviceB7));
}

TEST_F(DeviceTest, SyncCopyD2DSliceByCount) {
  EXPECT_NO_ERROR(
      Device.synchronousCopyD2D(DeviceA5, DeviceB7.asSlice().drop_front(2), 5));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], HostB7[I + 2]);
  }

  EXPECT_NO_ERROR(
      Device.synchronousCopyD2D(DeviceA7, DeviceB7.asSlice().drop_back(3), 2));
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostA7[I], HostB7[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA5, DeviceB5.asSlice(), 7));

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA7, DeviceB5.asSlice(), 7));

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA5, DeviceB7.asSlice(), 7));
}

TEST_F(DeviceTest, SyncCopyD2DSlice) {
  EXPECT_NO_ERROR(
      Device.synchronousCopyD2D(DeviceA5, DeviceB7.asSlice().drop_back(2)));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], HostB7[I]);
  }

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA7, DeviceB5.asSlice()));

  EXPECT_ERROR(Device.synchronousCopyD2D(DeviceA5, DeviceB7.asSlice()));
}

TEST_F(DeviceTest, SyncCopySliceD2DSliceByCount) {
  EXPECT_NO_ERROR(
      Device.synchronousCopyD2D(DeviceA5.asSlice(), DeviceB5.asSlice(), 5));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], HostB5[I]);
  }

  EXPECT_NO_ERROR(
      Device.synchronousCopyD2D(DeviceA7.asSlice(), DeviceB7.asSlice(), 2));
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostA7[I], HostB7[I]);
  }

  EXPECT_ERROR(
      Device.synchronousCopyD2D(DeviceA5.asSlice(), DeviceB5.asSlice(), 7));

  EXPECT_ERROR(
      Device.synchronousCopyD2D(DeviceA7.asSlice(), DeviceB5.asSlice(), 7));

  EXPECT_ERROR(
      Device.synchronousCopyD2D(DeviceA5.asSlice(), DeviceB7.asSlice(), 7));
}

TEST_F(DeviceTest, SyncCopySliceD2DSlice) {
  EXPECT_NO_ERROR(
      Device.synchronousCopyD2D(DeviceA5.asSlice(), DeviceB5.asSlice()));
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], HostB5[I]);
  }

  EXPECT_ERROR(
      Device.synchronousCopyD2D(DeviceA7.asSlice(), DeviceB5.asSlice()));

  EXPECT_ERROR(
      Device.synchronousCopyD2D(DeviceA5.asSlice(), DeviceB7.asSlice()));
}

} // namespace
