//===------- Offload API tests - olIterateCompatibleDevices -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olIterateCompatibleDevicesTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olIterateCompatibleDevicesTest);

TEST_P(olIterateCompatibleDevicesTest, Success) {
  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
  ASSERT_GE(DeviceBin->getBufferSize(), 0lu);

  struct CallbackDataTy {
    ol_device_handle_t ExpectedDevice;
    bool Found = false;
  } CallbackData{Device};

  ASSERT_SUCCESS(olIterateCompatibleDevices(
      DeviceBin->getBufferStart(), DeviceBin->getBufferSize(),
      [](ol_device_handle_t D, void *UserData) {
        auto *Data = static_cast<CallbackDataTy *>(UserData);
        if (D == Data->ExpectedDevice)
          Data->Found = true;
        return true;
      },
      &CallbackData));

  ASSERT_TRUE(CallbackData.Found);
}

TEST_P(olIterateCompatibleDevicesTest, SuccessStopIteration) {
  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
  ASSERT_GE(DeviceBin->getBufferSize(), 0lu);

  uint32_t CallCount = 0;
  ASSERT_SUCCESS(olIterateCompatibleDevices(
      DeviceBin->getBufferStart(), DeviceBin->getBufferSize(),
      [](ol_device_handle_t, void *UserData) {
        auto *Count = static_cast<uint32_t *>(UserData);
        *Count += 1;
        return false;
      },
      &CallCount));

  ASSERT_EQ(CallCount, 1u);
}

TEST_P(olIterateCompatibleDevicesTest, EmptyBinary) {
  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
  ASSERT_GE(DeviceBin->getBufferSize(), 0lu);

  uint32_t CallCount = 0;
  ASSERT_SUCCESS(olIterateCompatibleDevices(
      DeviceBin->getBufferStart(), 0,
      [](ol_device_handle_t, void *UserData) {
        auto *Count = static_cast<uint32_t *>(UserData);
        *Count += 1;
        return true;
      },
      &CallCount));

  ASSERT_EQ(CallCount, 0u);
}
