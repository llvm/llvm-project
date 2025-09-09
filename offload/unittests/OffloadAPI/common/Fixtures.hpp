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
#include <thread>

#include "Environment.hpp"

#pragma once

#ifndef ASSERT_SUCCESS
#define ASSERT_SUCCESS(ACTUAL)                                                 \
  do {                                                                         \
    ol_result_t Res = ACTUAL;                                                  \
    if (Res && Res->Code != OL_ERRC_SUCCESS) {                                 \
      GTEST_FAIL() << #ACTUAL " returned " << Res->Code << ": "                \
                   << Res->Details;                                            \
    }                                                                          \
  } while (0)
#endif

#ifndef ASSERT_SUCCESS_OR_UNSUPPORTED
#define ASSERT_SUCCESS_OR_UNSUPPORTED(ACTUAL)                                  \
  do {                                                                         \
    ol_result_t Res = ACTUAL;                                                  \
    if (Res && Res->Code == OL_ERRC_UNSUPPORTED) {                             \
      GTEST_SKIP() << #ACTUAL " returned unsupported; skipping test";          \
      return;                                                                  \
    } else if (Res && Res->Code != OL_ERRC_SUCCESS) {                          \
      GTEST_FAIL() << #ACTUAL " returned " << Res->Code << ": "                \
                   << Res->Details;                                            \
    }                                                                          \
  } while (0)
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

inline std::string SanitizeString(const std::string &Str) {
  auto NewStr = Str;
  std::replace_if(
      NewStr.begin(), NewStr.end(), [](char C) { return !std::isalnum(C); },
      '_');
  return NewStr;
}

template <typename Fn> inline void threadify(Fn body) {
  std::vector<std::thread> Threads;
  for (size_t I = 0; I < 20; I++) {
    Threads.emplace_back(
        [&body](size_t I) {
          std::string ScopeMsg{"Thread #"};
          ScopeMsg.append(std::to_string(I));
          SCOPED_TRACE(ScopeMsg);
          body(I);
        },
        I);
  }
  for (auto &T : Threads) {
    T.join();
  }
}

/// Enqueues a task to the queue that can be manually resolved.
// It will block until `trigger` is called.
struct ManuallyTriggeredTask {
  std::mutex M;
  std::condition_variable CV;
  bool Flag = false;
  ol_event_handle_t CompleteEvent;

  ol_result_t enqueue(ol_queue_handle_t Queue) {
    if (auto Err = olLaunchHostFunction(
            Queue,
            [](void *That) {
              static_cast<ManuallyTriggeredTask *>(That)->wait();
            },
            this))
      return Err;

    return olCreateEvent(Queue, &CompleteEvent);
  }

  void wait() {
    std::unique_lock<std::mutex> lk(M);
    CV.wait_for(lk, std::chrono::milliseconds(1000), [&] { return Flag; });
    EXPECT_TRUE(Flag);
  }

  ol_result_t trigger() {
    Flag = true;
    CV.notify_one();

    return olSyncEvent(CompleteEvent);
  }
};

struct OffloadTest : ::testing::Test {
  ol_device_handle_t Host = TestEnvironment::getHostDevice();
};

struct OffloadDeviceTest
    : OffloadTest,
      ::testing::WithParamInterface<TestEnvironment::Device> {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadTest::SetUp());

    auto DeviceParam = GetParam();
    Device = DeviceParam.Handle;
    if (Device == nullptr)
      GTEST_SKIP() << "No available devices.";
  }

  ol_platform_backend_t getPlatformBackend() const {
    ol_platform_handle_t Platform = nullptr;
    if (olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM,
                        sizeof(ol_platform_handle_t), &Platform))
      return OL_PLATFORM_BACKEND_UNKNOWN;
    ol_platform_backend_t Backend;
    if (olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND,
                          sizeof(ol_platform_backend_t), &Backend))
      return OL_PLATFORM_BACKEND_UNKNOWN;
    return Backend;
  }

  ol_device_handle_t Device = nullptr;
};

struct OffloadPlatformTest : OffloadDeviceTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::SetUp());

    ASSERT_SUCCESS(olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM,
                                   sizeof(Platform), &Platform));
    ASSERT_NE(Platform, nullptr);
  }

  ol_platform_handle_t Platform = nullptr;
};

// Fixture for a generic program test. If you want a different program, use
// offloadQueueTest and create your own program handle with the binary you want.
struct OffloadProgramTest : OffloadDeviceTest {
  void SetUp() override { SetUpWith("foo"); }

  void SetUpWith(const char *ProgramName) {
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::SetUp());
    ASSERT_TRUE(
        TestEnvironment::loadDeviceBinary(ProgramName, Device, DeviceBin));
    ASSERT_GE(DeviceBin->getBufferSize(), 0lu);
    ASSERT_SUCCESS(olCreateProgram(Device, DeviceBin->getBufferStart(),
                                   DeviceBin->getBufferSize(), &Program));
  }

  void TearDown() override {
    if (Program) {
      olDestroyProgram(Program);
    }
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::TearDown());
  }

  ol_program_handle_t Program = nullptr;
  std::unique_ptr<llvm::MemoryBuffer> DeviceBin;
};

struct OffloadKernelTest : OffloadProgramTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadProgramTest::SetUp());
    ASSERT_SUCCESS(olGetSymbol(Program, "foo", OL_SYMBOL_KIND_KERNEL, &Kernel));
  }

  void TearDown() override {
    RETURN_ON_FATAL_FAILURE(OffloadProgramTest::TearDown());
  }

  ol_symbol_handle_t Kernel = nullptr;
};

struct OffloadGlobalTest : OffloadProgramTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadProgramTest::SetUpWith("global"));
    ASSERT_SUCCESS(olGetSymbol(Program, "global",
                               OL_SYMBOL_KIND_GLOBAL_VARIABLE, &Global));
  }

  void TearDown() override {
    RETURN_ON_FATAL_FAILURE(OffloadProgramTest::TearDown());
  }

  ol_symbol_handle_t Global = nullptr;
};

struct OffloadQueueTest : OffloadDeviceTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::SetUp());
    ASSERT_SUCCESS(olCreateQueue(Device, &Queue));
  }

  void TearDown() override {
    if (Queue) {
      olDestroyQueue(Queue);
    }
    RETURN_ON_FATAL_FAILURE(OffloadDeviceTest::TearDown());
  }

  ol_queue_handle_t Queue = nullptr;
};

struct OffloadEventTest : OffloadQueueTest {
  void SetUp() override {
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::SetUp());
    ASSERT_SUCCESS(olCreateEvent(Queue, &Event));
    ASSERT_SUCCESS(olSyncQueue(Queue));
  }

  void TearDown() override {
    if (Event)
      olDestroyEvent(Event);
    RETURN_ON_FATAL_FAILURE(OffloadQueueTest::TearDown());
  }

  ol_event_handle_t Event = nullptr;
};

// Devices might not be available for offload testing, so allow uninstantiated
// tests (as the device list will be empty). This means that all tests requiring
// a device will be silently skipped.
#define OFFLOAD_TESTS_INSTANTIATE_DEVICE_FIXTURE(FIXTURE)                      \
  INSTANTIATE_TEST_SUITE_P(                                                    \
      , FIXTURE, ::testing::ValuesIn(TestEnvironment::getDevices()),           \
      [](const ::testing::TestParamInfo<TestEnvironment::Device> &info) {      \
        return SanitizeString(info.param.Name);                                \
      });                                                                      \
  GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(FIXTURE)
