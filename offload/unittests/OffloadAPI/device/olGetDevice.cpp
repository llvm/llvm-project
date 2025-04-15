//===------- Offload API tests - olGetDevice -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

// TODO: Rename
using olGetDeviceTest = OffloadTest;

TEST_F(olGetDeviceTest, Success) {
  uint32_t Count = 0;
  ASSERT_SUCCESS(olGetDeviceCount(&Count));
  if (Count == 0)
    GTEST_SKIP() << "No available devices.";

  std::vector<ol_device_handle_t> Devices(Count);
  ASSERT_SUCCESS(olGetDevices(Count, Devices.data()));
  for (auto Device : Devices) {
    ASSERT_NE(nullptr, Device);
  }
}

TEST_F(olGetDeviceTest, SuccessSubsetOfDevices) {
  uint32_t Count;
  ASSERT_SUCCESS(olGetDeviceCount(&Count));
  if (Count < 2)
    GTEST_SKIP() << "Only one device is available.";

  std::vector<ol_device_handle_t> Devices(Count - 1);
  ASSERT_SUCCESS(olGetDevices(Count - 1, Devices.data()));
  for (auto Device : Devices) {
    ASSERT_NE(nullptr, Device);
  }
}
