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

#include "SimpleHostPlatformDevice.h"
#include "streamexecutor/Device.h"
#include "streamexecutor/Kernel.h"
#include "streamexecutor/KernelSpec.h"
#include "streamexecutor/PlatformDevice.h"
#include "streamexecutor/Stream.h"

#include "gtest/gtest.h"

namespace {

namespace se = ::streamexecutor;

const auto &getDeviceValue =
    se::test::SimpleHostPlatformDevice::getDeviceValue<int>;

/// Test fixture to hold objects used by tests.
class StreamTest : public ::testing::Test {
public:
  StreamTest()
      : DummyPlatformStream(1), Device(&PDevice),
        Stream(&PDevice, &DummyPlatformStream), HostA5{0, 1, 2, 3, 4},
        HostB5{5, 6, 7, 8, 9}, HostA7{10, 11, 12, 13, 14, 15, 16},
        HostB7{17, 18, 19, 20, 21, 22, 23}, Host5{24, 25, 26, 27, 28},
        Host7{29, 30, 31, 32, 33, 34, 35},
        RegisteredHost5(getOrDie(
            Device.registerHostMemory(llvm::MutableArrayRef<int>(Host5)))),
        RegisteredHost7(getOrDie(
            Device.registerHostMemory(llvm::MutableArrayRef<int>(Host7)))),
        DeviceA5(getOrDie(Device.allocateDeviceMemory<int>(5))),
        DeviceB5(getOrDie(Device.allocateDeviceMemory<int>(5))),
        DeviceA7(getOrDie(Device.allocateDeviceMemory<int>(7))),
        DeviceB7(getOrDie(Device.allocateDeviceMemory<int>(7))) {
    se::dieIfError(Device.synchronousCopyH2D<int>(HostA5, DeviceA5));
    se::dieIfError(Device.synchronousCopyH2D<int>(HostB5, DeviceB5));
    se::dieIfError(Device.synchronousCopyH2D<int>(HostA7, DeviceA7));
    se::dieIfError(Device.synchronousCopyH2D<int>(HostB7, DeviceB7));
  }

protected:
  int DummyPlatformStream; // Mimicking a platform where the platform stream
                           // handle is just a stream number.
  se::test::SimpleHostPlatformDevice PDevice;
  se::Device Device;
  se::Stream Stream;

  // Device memory is matched by host arrays.
  int HostA5[5];
  int HostB5[5];
  int HostA7[7];
  int HostB7[7];

  // Host memory to be used as actual host memory.
  int Host5[5];
  int Host7[7];

  se::RegisteredHostMemory<int> RegisteredHost5;
  se::RegisteredHostMemory<int> RegisteredHost7;

  // Device memory.
  se::GlobalDeviceMemory<int> DeviceA5;
  se::GlobalDeviceMemory<int> DeviceB5;
  se::GlobalDeviceMemory<int> DeviceA7;
  se::GlobalDeviceMemory<int> DeviceB7;
};

using llvm::ArrayRef;
using llvm::MutableArrayRef;

// D2H tests

TEST_F(StreamTest, CopyD2HToRegisteredRefByCount) {
  Stream.thenCopyD2H(DeviceA5, RegisteredHost5, 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceB5, RegisteredHost5, 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostB5[I], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceA7, RegisteredHost5, 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2HToRegistered) {
  Stream.thenCopyD2H(DeviceA5, RegisteredHost5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceA5, RegisteredHost7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2HSliceToRegiseredSliceByCount) {
  Stream.thenCopyD2H(DeviceA5.asSlice().slice(1),
                     RegisteredHost5.asSlice().slice(1, 4), 4);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 1; I < 5; ++I) {
    EXPECT_EQ(HostA5[I], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceB5.asSlice().drop_back(1), RegisteredHost5, 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(HostB5[I], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceA5.asSlice(), RegisteredHost7, 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2HSliceToRegistered) {
  Stream.thenCopyD2H(DeviceA7.asSlice().slice(1, 5), RegisteredHost5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(HostA7[I + 1], Host5[I]);
  }

  Stream.thenCopyD2H(DeviceA5.asSlice(), RegisteredHost7);
  EXPECT_FALSE(Stream.isOK());
}

// H2D tests

TEST_F(StreamTest, CopyH2DFromRegisterdByCount) {
  Stream.thenCopyH2D(RegisteredHost5, DeviceA5, 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA5, I), Host5[I]);
  }

  Stream.thenCopyH2D(RegisteredHost5, DeviceB5, 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceB5, I), Host5[I]);
  }

  Stream.thenCopyH2D(RegisteredHost7, DeviceA5, 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyH2DFromRegistered) {
  Stream.thenCopyH2D(RegisteredHost5, DeviceA5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA5, I), Host5[I]);
  }

  Stream.thenCopyH2D(RegisteredHost7, DeviceA5);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyH2DFromRegisteredSliceToSlice) {
  Stream.thenCopyH2D(RegisteredHost5.asSlice().slice(1, 4),
                     DeviceA5.asSlice().slice(1), 4);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 1; I < 5; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA5, I), Host5[I]);
  }

  Stream.thenCopyH2D(RegisteredHost5, DeviceB5.asSlice().drop_back(1), 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceB5, I), Host5[I]);
  }

  Stream.thenCopyH2D(RegisteredHost5, DeviceA5.asSlice(), 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyH2DRegisteredToSlice) {
  Stream.thenCopyH2D(RegisteredHost5, DeviceA5.asSlice());
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA5, I), Host5[I]);
  }

  Stream.thenCopyH2D(RegisteredHost7, DeviceA5.asSlice());
  EXPECT_FALSE(Stream.isOK());
}

// D2D tests

TEST_F(StreamTest, CopyD2DByCount) {
  Stream.thenCopyD2D(DeviceA5, DeviceB5, 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA5, I), getDeviceValue(DeviceB5, I));
  }

  Stream.thenCopyD2D(DeviceA7, DeviceB7, 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA7, I), getDeviceValue(DeviceB7, I));
  }

  Stream.thenCopyD2D(DeviceA7, DeviceB5, 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2D) {
  Stream.thenCopyD2D(DeviceA5, DeviceB5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA5, I), getDeviceValue(DeviceB5, I));
  }

  Stream.thenCopyD2D(DeviceA7, DeviceB5);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopySliceD2DByCount) {
  Stream.thenCopyD2D(DeviceA5.asSlice().slice(1), DeviceB5, 4);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 4; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA5, I + 1), getDeviceValue(DeviceB5, I));
  }

  Stream.thenCopyD2D(DeviceA7.asSlice().drop_back(1), DeviceB7, 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA7, I), getDeviceValue(DeviceB7, I));
  }

  Stream.thenCopyD2D(DeviceA5.asSlice(), DeviceB5, 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopySliceD2D) {
  Stream.thenCopyD2D(DeviceA7.asSlice().drop_back(2), DeviceB5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA7, I), getDeviceValue(DeviceB5, I));
  }

  Stream.thenCopyD2D(DeviceA5.asSlice().drop_back(1), DeviceB7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2DSliceByCount) {
  Stream.thenCopyD2D(DeviceA5, DeviceB7.asSlice().slice(2), 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA5, I), getDeviceValue(DeviceB7, I + 2));
  }

  Stream.thenCopyD2D(DeviceA7, DeviceB7.asSlice().drop_back(3), 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA7, I), getDeviceValue(DeviceB7, I));
  }

  Stream.thenCopyD2D(DeviceA5, DeviceB7.asSlice(), 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopyD2DSlice) {
  Stream.thenCopyD2D(DeviceA5, DeviceB7.asSlice().drop_back(2));
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA5, I), getDeviceValue(DeviceB7, I));
  }

  Stream.thenCopyD2D(DeviceA5, DeviceB7.asSlice());
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopySliceD2DSliceByCount) {
  Stream.thenCopyD2D(DeviceA5.asSlice(), DeviceB5.asSlice(), 5);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA5, I), getDeviceValue(DeviceB5, I));
  }

  Stream.thenCopyD2D(DeviceA7.asSlice(), DeviceB7.asSlice(), 2);
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 2; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA7, I), getDeviceValue(DeviceB7, I));
  }

  Stream.thenCopyD2D(DeviceA7.asSlice(), DeviceB5.asSlice(), 7);
  EXPECT_FALSE(Stream.isOK());
}

TEST_F(StreamTest, CopySliceD2DSlice) {
  Stream.thenCopyD2D(DeviceA5.asSlice(), DeviceB5.asSlice());
  EXPECT_TRUE(Stream.isOK());
  for (int I = 0; I < 5; ++I) {
    EXPECT_EQ(getDeviceValue(DeviceA5, I), getDeviceValue(DeviceB5, I));
  }

  Stream.thenCopyD2D(DeviceA5.asSlice(), DeviceB7.asSlice());
  EXPECT_FALSE(Stream.isOK());
}

} // namespace
