//===------- Offload API tests - olInit -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// NOTE: For this test suite, the implicit olInit/olShutDown doesn't happen, so
// tests have to do it themselves

#include "../common/Fixtures.hpp"
#include <OffloadAPI.h>
#include <gtest/gtest.h>

struct olInitTest : ::testing::Test {};

TEST_F(olInitTest, Success) {
  ASSERT_SUCCESS(olInit());
  ASSERT_SUCCESS(olShutDown());
}

TEST_F(olInitTest, Uninitialized) {
  ASSERT_ERROR(OL_ERRC_UNINITIALIZED,
               olIterateDevices(
                   [](ol_device_handle_t, void *) { return false; }, nullptr));
}

TEST_F(olInitTest, RepeatedInit) {
  for (size_t I = 0; I < 10; I++) {
    ASSERT_SUCCESS(olInit());
    ASSERT_SUCCESS(olShutDown());
  }
}
