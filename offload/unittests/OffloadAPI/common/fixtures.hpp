//===------- Offload API tests - gtest fixtures --==-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <offload_api.h>
#include <offload_print.hpp>

#include "environment.hpp"

#pragma once

#ifndef ASSERT_SUCCESS
#define ASSERT_SUCCESS(ACTUAL) ASSERT_EQ(OFFLOAD_SUCCESS, ACTUAL)
#endif

// TODO: rework this so the EXPECTED/ACTUAL results are readable
#ifndef ASSERT_ERROR
#define ASSERT_ERROR(EXPECTED, ACTUAL)                                         \
  do {                                                                         \
    offload_result_t Res = ACTUAL;                                             \
    ASSERT_TRUE(Res && (Res->code == EXPECTED));                               \
  } while (0)
#endif

#define RETURN_ON_FATAL_FAILURE(...)                                           \
  __VA_ARGS__;                                                                 \
  if (this->HasFatalFailure() || this->IsSkipped()) {                          \
    return;                                                                    \
  }                                                                            \
  (void)0

struct offloadPlatformTest : ::testing::Test {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(::testing::Test::SetUp());

    Platform = TestEnvironment::getPlatform();
    ASSERT_NE(Platform, nullptr);
  }

  offload_platform_handle_t Platform;
};

struct offloadDeviceTest : offloadPlatformTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(offloadPlatformTest::SetUp());

    uint32_t NumDevices;
    ASSERT_SUCCESS(offloadDeviceGet(Platform, OFFLOAD_DEVICE_TYPE_ALL, 0,
                                    nullptr, &NumDevices));
    if (NumDevices == 0) {
      GTEST_SKIP() << "No available devices on this platform.";
    }
    ASSERT_SUCCESS(offloadDeviceGet(Platform, OFFLOAD_DEVICE_TYPE_ALL, 1,
                                    &Device, nullptr));
  }

  offload_device_handle_t Device;
};
