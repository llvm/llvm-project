//===------- Offload API tests - olIsValidBinary --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olIsValidBinaryTest = OffloadDeviceTest;
OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(olIsValidBinaryTest);

TEST_P(olIsValidBinaryTest, Success) {

  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
  ASSERT_GE(DeviceBin->getBufferSize(), 0lu);

  bool IsValid = false;
  ASSERT_SUCCESS(olIsValidBinary(Device, DeviceBin->getBufferStart(),
                                 DeviceBin->getBufferSize(), &IsValid));
  ASSERT_TRUE(IsValid);

  ASSERT_SUCCESS(
      olIsValidBinary(Device, DeviceBin->getBufferStart(), 0, &IsValid));
  ASSERT_FALSE(IsValid);
}

TEST_P(olIsValidBinaryTest, Invalid) {

  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
  ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Device, DeviceBin));
  ASSERT_GE(DeviceBin->getBufferSize(), 0lu);

  bool IsValid = false;
  ASSERT_SUCCESS(
      olIsValidBinary(Device, DeviceBin->getBufferStart(), 0, &IsValid));
  ASSERT_FALSE(IsValid);
}

TEST_P(olIsValidBinaryTest, NullPointer) {
  bool IsValid = false;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER,
               olIsValidBinary(Device, nullptr, 42, &IsValid));
  ASSERT_FALSE(IsValid);
}
