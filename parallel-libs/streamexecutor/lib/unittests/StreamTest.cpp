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

#include "streamexecutor/Device.h"
#include "streamexecutor/Kernel.h"
#include "streamexecutor/KernelSpec.h"
#include "streamexecutor/PlatformInterfaces.h"
#include "streamexecutor/Stream.h"

#include "gtest/gtest.h"

namespace {

namespace se = ::streamexecutor;

/// Mock PlatformDevice that performs asynchronous memcpy operations by
/// ignoring the stream argument and calling std::memcpy on device memory
/// handles.
class MockPlatformDevice : public se::PlatformDevice {
public:
  ~MockPlatformDevice() override {}

  std::string getName() const override { return "MockPlatformDevice"; }

  se::Expected<std::unique_ptr<se::PlatformStreamHandle>>
  createStream() override {
    return nullptr;
  }

  se::Error copyD2H(se::PlatformStreamHandle *S,
                    const se::GlobalDeviceMemoryBase &DeviceSrc,
                    size_t SrcByteOffset, void *HostDst, size_t DstByteOffset,
                    size_t ByteCount) override {
    std::memcpy(HostDst, static_cast<const char *>(DeviceSrc.getHandle()) +
                             SrcByteOffset,
                ByteCount);
    return se::Error::success();
  }

  se::Error copyH2D(se::PlatformStreamHandle *S, const void *HostSrc,
                    size_t SrcByteOffset, se::GlobalDeviceMemoryBase DeviceDst,
                    size_t DstByteOffset, size_t ByteCount) override {
    std::memcpy(static_cast<char *>(const_cast<void *>(DeviceDst.getHandle())) +
                    DstByteOffset,
                HostSrc, ByteCount);
    return se::Error::success();
  }

  se::Error copyD2D(se::PlatformStreamHandle *S,
                    const se::GlobalDeviceMemoryBase &DeviceSrc,
                    size_t SrcByteOffset, se::GlobalDeviceMemoryBase DeviceDst,
                    size_t DstByteOffset, size_t ByteCount) override {
    std::memcpy(static_cast<char *>(const_cast<void *>(DeviceDst.getHandle())) +
                    DstByteOffset,
                static_cast<const char *>(DeviceSrc.getHandle()) +
                    SrcByteOffset,
                ByteCount);
    return se::Error::success();
  }
};

/// Test fixture to hold objects used by tests.
class StreamTest : public ::testing::Test {
public:
  StreamTest()
      : HostA5{0, 1, 2, 3, 4}, HostB5{5, 6, 7, 8, 9},
        HostA7{10, 11, 12, 13, 14, 15, 16}, HostB7{17, 18, 19, 20, 21, 22, 23},
        DeviceA5(se::GlobalDeviceMemory<int>::makeFromElementCount(HostA5, 5)),
        DeviceB5(se::GlobalDeviceMemory<int>::makeFromElementCount(HostB5, 5)),
        DeviceA7(se::GlobalDeviceMemory<int>::makeFromElementCount(HostA7, 7)),
        DeviceB7(se::GlobalDeviceMemory<int>::makeFromElementCount(HostB7, 7)),
        Host5{24, 25, 26, 27, 28}, Host7{29, 30, 31, 32, 33, 34, 35},
        Stream(llvm::make_unique<se::PlatformStreamHandle>(&PDevice)) {}

protected:
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
  se::Stream Stream;
};

using llvm::ArrayRef;
using llvm::MutableArrayRef;

// D2H tests

TEST_F(StreamTest, CopyD2HToMutableArrayRefByCount) {
  Stream.thenCopyD2H(DeviceA5, MutableArrayRef<int>(Host5), 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceB5, MutableArrayRef<int>(Host5), 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostB5[I], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceA7, MutableArrayRef<int>(Host5), 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2HToMutableArrayRef) {
  Stream.thenCopyD2H(DeviceA5, MutableArrayRef<int>(Host5));
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceA5, MutableArrayRef<int>(Host7));
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2HToPointer) {
  Stream.thenCopyD2H(DeviceA5, Host5, 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceA5, Host7, 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2HSliceToMutableArrayRefByCount) {
  Stream.thenCopyD2H(DeviceA5.asSlice().drop_front(1),
                     MutableArrayRef<int>(Host5 + 1, 4), 4);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 1; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceB5.asSlice().drop_back(1),
                     MutableArrayRef<int>(Host5), 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostB5[I], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceA5.asSlice(), MutableArrayRef<int>(Host7), 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2HSliceToMutableArrayRef) {
  Stream.thenCopyD2H(DeviceA7.asSlice().slice(1, 5),
                     MutableArrayRef<int>(Host5));
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA7[I + 1], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceA5.asSlice(), MutableArrayRef<int>(Host7));
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2HSliceToPointer) {
  Stream.thenCopyD2H(DeviceA5.asSlice().drop_front(1), Host5 + 1, 4);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 1; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceA5.asSlice(), Host7, 7);
  EXPECT_FALSE(Stream.isOK());
}

// H2D tests

TEST_F(StreamTest, CopyH2DToArrayRefByCount) {
  Stream.thenCopyH2D(ArrayRef<int>(Host5), DeviceA5, 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyH2D(ArrayRef<int>(Host5), DeviceB5, 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostB5[I], Host5[I]);
  }

  Stream.thenCopyH2D(ArrayRef<int>(Host7), DeviceA5, 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyH2DToArrayRef) {
  Stream.thenCopyH2D(ArrayRef<int>(Host5), DeviceA5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyH2D(ArrayRef<int>(Host7), DeviceA5);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyH2DToPointer) {
  Stream.thenCopyH2D(Host5, DeviceA5, 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyH2D(Host7, DeviceA5, 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyH2DSliceToArrayRefByCount) {
  Stream.thenCopyH2D(ArrayRef<int>(Host5 + 1, 4),
                     DeviceA5.asSlice().drop_front(1), 4);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 1; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyH2D(ArrayRef<int>(Host5), DeviceB5.asSlice().drop_back(1), 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostB5[I], Host5[I]);
  }

  Stream.thenCopyH2D(ArrayRef<int>(Host5), DeviceA5.asSlice(), 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyH2DSliceToArrayRef) {

  Stream.thenCopyH2D(ArrayRef<int>(Host5), DeviceA5.asSlice());
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyH2D(ArrayRef<int>(Host7), DeviceA5.asSlice());
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyH2DSliceToPointer) {
  Stream.thenCopyH2D(Host5, DeviceA5.asSlice(), 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyH2D(Host7, DeviceA5.asSlice(), 7);
  EXPECT_FALSE(Stream.isOK());
}

// D2D tests

TEST_F(StreamTest, CopyD2DByCount) {
  Stream.thenCopyD2D(DeviceA5, DeviceB5, 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], HostB5[I]);
  }

  Stream.thenCopyD2D(DeviceA7, DeviceB7, 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostA7[I], HostB7[I]);
  }

  Stream.thenCopyD2D(DeviceA7, DeviceB5, 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2D) {
  Stream.thenCopyD2D(DeviceA5, DeviceB5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], HostB5[I]);
  }

  Stream.thenCopyD2D(DeviceA7, DeviceB5);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopySliceD2DByCount) {
  Stream.thenCopyD2D(DeviceA5.asSlice().drop_front(1), DeviceB5, 4);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 4; ++I) {
    EXPECT_EQ(HostA5[I + 1], HostB5[I]);
  }

  Stream.thenCopyD2D(DeviceA7.asSlice().drop_back(1), DeviceB7, 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostA7[I], HostB7[I]);
  }

  Stream.thenCopyD2D(DeviceA5.asSlice(), DeviceB5, 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopySliceD2D) {

  Stream.thenCopyD2D(DeviceA7.asSlice().drop_back(2), DeviceB5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA7[I], HostB5[I]);
  }

  Stream.thenCopyD2D(DeviceA5.asSlice().drop_back(1), DeviceB7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2DSliceByCount) {
  Stream.thenCopyD2D(DeviceA5, DeviceB7.asSlice().drop_front(2), 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], HostB7[I + 2]);
  }

  Stream.thenCopyD2D(DeviceA7, DeviceB7.asSlice().drop_back(3), 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostA7[I], HostB7[I]);
  }

  Stream.thenCopyD2D(DeviceA5, DeviceB7.asSlice(), 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2DSlice) {

  Stream.thenCopyD2D(DeviceA5, DeviceB7.asSlice().drop_back(2));
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], HostB7[I]);
  }

  Stream.thenCopyD2D(DeviceA5, DeviceB7.asSlice());
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopySliceD2DSliceByCount) {

  Stream.thenCopyD2D(DeviceA5.asSlice(), DeviceB5.asSlice(), 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], HostB5[I]);
  }

  Stream.thenCopyD2D(DeviceA7.asSlice(), DeviceB7.asSlice(), 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostA7[I], HostB7[I]);
  }

  Stream.thenCopyD2D(DeviceA7.asSlice(), DeviceB5.asSlice(), 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopySliceD2DSlice) {

  Stream.thenCopyD2D(DeviceA5.asSlice(), DeviceB5.asSlice());
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], HostB5[I]);
  }

  Stream.thenCopyD2D(DeviceA5.asSlice(), DeviceB7.asSlice());
  EXPECT_FALSE(Stream.isOK());
}

} // namespace
