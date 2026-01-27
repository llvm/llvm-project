//===------- Offload API tests - olGetHostDevice -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

using olGetHostDeviceTest = OffloadTest;

TEST_F(olGetHostDeviceTest, SuccessGetHostDevice) {
  ol_device_handle_t Host = nullptr;
  ASSERT_SUCCESS(olGetHostDevice(&Host));
  ASSERT_NE(Host, nullptr);
}
