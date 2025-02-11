//===------- Offload API tests - gtest fixtures --==-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <OffloadAPI.h>
#include <OffloadPrint.hpp>
#include <gtest/gtest.h>

#include "Environment.hpp"

#pragma once

#ifndef ASSERT_SUCCESS
#define ASSERT_SUCCESS(ACTUAL) ASSERT_EQ(OL_SUCCESS, ACTUAL)
#endif

// TODO: rework this so the EXPECTED/ACTUAL results are readable
#ifndef ASSERT_ERROR
#define ASSERT_ERROR(EXPECTED, ACTUAL)                                         \
  do {                                                                         \
    ol_result_t Res = ACTUAL;                                                  \
    ASSERT_TRUE(Res && (Res->Code == EXPECTED));                               \
  } while (0)
#endif

#ifndef ASSERT_ANY_ERROR
#define ASSERT_ANY_ERROR(ACTUAL)                                               \
  do {                                                                         \
    ol_result_t Res = ACTUAL;                                                  \
    ASSERT_TRUE(Res);                                                          \
  } while (0)
#endif

#define RETURN_ON_FATAL_FAILURE(...)                                           \
  __VA_ARGS__;                                                                 \
  if (this->HasFatalFailure() || this->IsSkipped()) {                          \
    return;                                                                    \
  }                                                                            \
  (void)0

struct offloadTest : ::testing::Test {
  // No special behavior now, but just in case we need to override it in future
};

struct offloadPlatformTest : offloadTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(offloadTest::SetUp());

    Platform = TestEnvironment::getPlatform();
    ASSERT_NE(Platform, nullptr);
  }

  ol_platform_handle_t Platform;
};

struct offloadDeviceTest : offloadPlatformTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(offloadPlatformTest::SetUp());

    uint32_t NumDevices;
    ASSERT_SUCCESS(olGetDeviceCount(Platform, &NumDevices));
    if (NumDevices == 0)
      GTEST_SKIP() << "No available devices on this platform.";
    ASSERT_SUCCESS(olGetDevice(Platform, 1, &Device));
  }

  ol_device_handle_t Device = nullptr;
};

// Fixture for a generic program test. If you want a different program, use
// offloadQueueTest and create your own program handle with the binary you want.
struct offloadProgramTest : offloadDeviceTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(offloadDeviceTest::SetUp());
    ASSERT_TRUE(TestEnvironment::loadDeviceBinary("foo", Platform, DeviceBin));
    ASSERT_GE(DeviceBin->size(), 0lu);
    ASSERT_SUCCESS(olCreateProgram(Device, DeviceBin->data(), DeviceBin->size(),
                                   &Program));
  }

  void TearDown() override {
    if (Program) {
      olReleaseProgram(Program);
    }
    RETURN_ON_FATAL_FAILURE(offloadDeviceTest::TearDown());
  }

  ol_program_handle_t Program = nullptr;
  std::shared_ptr<std::vector<char>> DeviceBin;
};

struct offloadKernelTest : offloadProgramTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(offloadProgramTest::SetUp());
    ASSERT_SUCCESS(olCreateKernel(Program, "foo", &Kernel));
  }

  void TearDown() override {
    if (Kernel) {
      olReleaseKernel(Kernel);
    }
    RETURN_ON_FATAL_FAILURE(offloadProgramTest::TearDown());
  }

  ol_kernel_handle_t Kernel = nullptr;
};

struct offloadQueueTest : offloadDeviceTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(offloadDeviceTest::SetUp());
    ASSERT_SUCCESS(olCreateQueue(Device, &Queue));
  }

  void TearDown() override {
    if (Queue) {
      olReleaseQueue(Queue);
    }
    RETURN_ON_FATAL_FAILURE(offloadDeviceTest::TearDown());
  }

  ol_queue_handle_t Queue = nullptr;
};
