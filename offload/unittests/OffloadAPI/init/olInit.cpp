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
#include "llvm/ADT/STLExtras.h"
#include <OffloadAPI.h>
#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

struct olInitTest : ::testing::Test {};

static void setEnv(const char *Name, const char *Value) {
#ifdef _WIN32
  _putenv_s(Name, Value);
#else
  setenv(Name, Value, 1);
#endif
}

static void unsetEnv(const char *Name) {
#ifdef _WIN32
  _putenv_s(Name, "");
#else
  unsetenv(Name);
#endif
}

// Collects the backend of every usable platform.
static std::vector<ol_platform_backend_t> getBackends() {
  std::vector<ol_platform_backend_t> Backends;
  olIteratePlatforms(
      [](ol_platform_handle_t Platform, void *Data) {
        ol_platform_backend_t Backend;
        olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND, sizeof(Backend),
                          &Backend);
        static_cast<std::vector<ol_platform_backend_t> *>(Data)->push_back(
            Backend);
        return true;
      },
      &Backends);
  std::sort(Backends.begin(), Backends.end());
  return Backends;
}

TEST_F(olInitTest, Success) {
  ASSERT_SUCCESS(olInit(nullptr));
  ASSERT_SUCCESS(olShutDown());
}

// OlInit must be called before any other function. The others document
// OL_ERRC_UNINITIALIZED as their return value for this case.
TEST_F(olInitTest, Uninitialized) {
  ASSERT_ERROR(OL_ERRC_UNINITIALIZED,
               olIterateDevices(
                   [](ol_device_handle_t, void *) { return false; }, nullptr));
}

// OlShutDown is no exception; before any olInit it also reports
// OL_ERRC_UNINITIALIZED. Unlike every other entry point, this check is not
// emitted by offload-tblgen but done inside olShutDown itself, so it needs its
// own test.
TEST_F(olInitTest, ShutDownUninitialized) {
  ASSERT_ERROR(OL_ERRC_UNINITIALIZED, olShutDown());
}

TEST_F(olInitTest, RepeatedInit) {
  for (size_t I = 0; I < 10; I++) {
    ASSERT_SUCCESS(olInit(nullptr));
    ASSERT_SUCCESS(olShutDown());
  }
}

TEST_F(olInitTest, RepeatedInitWithDevices) {
  size_t Expected = 0;
  for (size_t I = 0; I < 3; I++) {
    ASSERT_SUCCESS(olInit(nullptr));
    size_t Count = 0;

    // Force the platform to be initialized.
    ASSERT_SUCCESS(olIterateDevices(
        [](ol_device_handle_t, void *Data) {
          ++*static_cast<size_t *>(Data);
          return true;
        },
        &Count));
    ASSERT_SUCCESS(olShutDown());
    if (I == 0)
      Expected = Count;
    ASSERT_EQ(Count, Expected);
  }
}

// Each olInit adds a reference and the library is only torn down once
// every reference has been released by olShutDown. Initializing again after a
// complete shutdown is allowed (see RepeatedInit).
TEST_F(olInitTest, ReferenceCount) {
  constexpr size_t Depth = 5;
  for (size_t I = 0; I < Depth; I++)
    ASSERT_SUCCESS(olInit(nullptr));

  for (size_t I = 0; I < Depth - 1; I++) {
    ASSERT_SUCCESS(olShutDown());
    ASSERT_SUCCESS(olIteratePlatforms(
        [](ol_platform_handle_t, void *) { return false; }, nullptr));
  }

  ASSERT_SUCCESS(olShutDown());
  ASSERT_ERROR(
      OL_ERRC_UNINITIALIZED,
      olIteratePlatforms([](ol_platform_handle_t, void *) { return false; },
                         nullptr));
  ASSERT_ERROR(OL_ERRC_UNINITIALIZED, olShutDown());
}

TEST_F(olInitTest, WithInitArgs) {
  ol_init_args_t Args = OL_INIT_ARGS_INIT;
  ol_platform_backend_t Backends[] = {OL_PLATFORM_BACKEND_HOST};
  Args.NumPlatforms = 1;
  Args.Platforms = Backends;
  ASSERT_SUCCESS(olInit(&Args));
  ASSERT_SUCCESS(olShutDown());
}

// NULL InitArgs and NumPlatforms == 0 both select every platform.
TEST_F(olInitTest, DefaultSelectsAllPlatforms) {
  ASSERT_SUCCESS(olInit(nullptr));
  auto Default = getBackends();
  ASSERT_SUCCESS(olShutDown());

  ol_init_args_t Args = OL_INIT_ARGS_INIT;
  ASSERT_SUCCESS(olInit(&Args));
  auto Explicit = getBackends();
  ASSERT_SUCCESS(olShutDown());

  ASSERT_FALSE(Default.empty());
  ASSERT_TRUE(llvm::is_contained(Default, OL_PLATFORM_BACKEND_HOST));
  ASSERT_EQ(Default, Explicit);
}

// Only the listed backends are usable and the host is always usable.
TEST_F(olInitTest, SelectHostOnly) {
  ol_init_args_t Args = OL_INIT_ARGS_INIT;
  ol_platform_backend_t Backends[] = {OL_PLATFORM_BACKEND_HOST};
  Args.NumPlatforms = 1;
  Args.Platforms = Backends;
  ASSERT_SUCCESS(olInit(&Args));
  std::vector<ol_platform_backend_t> Expected = {OL_PLATFORM_BACKEND_HOST};
  ASSERT_EQ(getBackends(), Expected);
  ASSERT_SUCCESS(olShutDown());
}

// The host platform is usable even when it is not requested.
TEST_F(olInitTest, HostAlwaysSelected) {
  ol_init_args_t Args = OL_INIT_ARGS_INIT;
  ol_platform_backend_t Backends[] = {OL_PLATFORM_BACKEND_CUDA};
  Args.NumPlatforms = 1;
  Args.Platforms = Backends;
  ASSERT_SUCCESS(olInit(&Args));
  auto Selected = getBackends();
  ASSERT_TRUE(llvm::is_contained(Selected, OL_PLATFORM_BACKEND_HOST));
  for (auto Backend : Selected)
    ASSERT_TRUE(Backend == OL_PLATFORM_BACKEND_HOST ||
                Backend == OL_PLATFORM_BACKEND_CUDA);
  ASSERT_SUCCESS(olShutDown());
}

// Duplicate entries are ignored without error.
TEST_F(olInitTest, DuplicateBackends) {
  ol_init_args_t Args = OL_INIT_ARGS_INIT;
  ol_platform_backend_t Backends[] = {OL_PLATFORM_BACKEND_HOST,
                                      OL_PLATFORM_BACKEND_HOST,
                                      OL_PLATFORM_BACKEND_HOST};
  Args.NumPlatforms = 3;
  Args.Platforms = Backends;
  ASSERT_SUCCESS(olInit(&Args));
  std::vector<ol_platform_backend_t> Expected = {OL_PLATFORM_BACKEND_HOST};
  ASSERT_EQ(getBackends(), Expected);
  ASSERT_SUCCESS(olShutDown());
}

// Unknown backends are ignored without error.
TEST_F(olInitTest, UnknownBackends) {
  ol_init_args_t Args = OL_INIT_ARGS_INIT;
  ol_platform_backend_t Backends[] = {
      OL_PLATFORM_BACKEND_UNKNOWN, static_cast<ol_platform_backend_t>(0x1234)};
  Args.NumPlatforms = 2;
  Args.Platforms = Backends;
  ASSERT_SUCCESS(olInit(&Args));
  std::vector<ol_platform_backend_t> Expected = {OL_PLATFORM_BACKEND_HOST};
  ASSERT_EQ(getBackends(), Expected);
  ASSERT_SUCCESS(olShutDown());
}

// InitArgs is applied once, by the call that brings the count from zero
// to one. Nested calls ignore it entirely, even when it would be invalid.
TEST_F(olInitTest, NestedInitIgnoresArgs) {
  ol_init_args_t HostOnly = OL_INIT_ARGS_INIT;
  ol_platform_backend_t Backends[] = {OL_PLATFORM_BACKEND_HOST};
  HostOnly.NumPlatforms = 1;
  HostOnly.Platforms = Backends;
  ASSERT_SUCCESS(olInit(&HostOnly));
  auto Selected = getBackends();

  ASSERT_SUCCESS(olInit(nullptr));
  ASSERT_EQ(getBackends(), Selected);

  ol_init_args_t Invalid = OL_INIT_ARGS_INIT;
  Invalid.Size = 0;
  ASSERT_SUCCESS(olInit(&Invalid));
  ASSERT_EQ(getBackends(), Selected);

  ASSERT_SUCCESS(olShutDown());
  ASSERT_SUCCESS(olShutDown());
  ASSERT_SUCCESS(olShutDown());
}

// Platforms is only read during olInit and may be freed afterwards.
TEST_F(olInitTest, PlatformsArrayNotRetained) {
  auto *Backends = new ol_platform_backend_t[1]{OL_PLATFORM_BACKEND_HOST};
  ol_init_args_t Args = OL_INIT_ARGS_INIT;
  Args.NumPlatforms = 1;
  Args.Platforms = Backends;
  ASSERT_SUCCESS(olInit(&Args));
  delete[] Backends;

  std::vector<ol_platform_backend_t> Expected = {OL_PLATFORM_BACKEND_HOST};
  ASSERT_EQ(getBackends(), Expected);
  ASSERT_SUCCESS(olShutDown());
}

// Size smaller than sizeof(ol_init_args_t) is rejected.
TEST_F(olInitTest, InvalidSize) {
  ol_init_args_t Args = OL_INIT_ARGS_INIT;
  Args.Size = 0;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE, olInit(&Args));

  Args.Size = sizeof(ol_init_args_t) - 1;
  ASSERT_ERROR(OL_ERRC_INVALID_SIZE, olInit(&Args));
  ASSERT_ERROR(OL_ERRC_UNINITIALIZED, olShutDown());
}

// A larger Size is accepted for forward compatibility.
TEST_F(olInitTest, LargerSizeAccepted) {
  ol_init_args_t Args = OL_INIT_ARGS_INIT;
  Args.Size = sizeof(ol_init_args_t) + 64;
  ASSERT_SUCCESS(olInit(&Args));
  ASSERT_SUCCESS(olShutDown());
}

// NumPlatforms > 0 with a NULL Platforms array is rejected.
TEST_F(olInitTest, NullPlatformsArray) {
  ol_init_args_t Args = OL_INIT_ARGS_INIT;
  Args.NumPlatforms = 1;
  Args.Platforms = nullptr;
  ASSERT_ERROR(OL_ERRC_INVALID_NULL_POINTER, olInit(&Args));
  ASSERT_ERROR(OL_ERRC_UNINITIALIZED, olShutDown());
}

// Environment variables are read during olInit only. Changing
// OFFLOAD_TRACE afterwards has no effect until the library is reinitialized.
TEST_F(olInitTest, TraceReadAtInit) {
  unsetEnv("OFFLOAD_TRACE");
  ASSERT_SUCCESS(olInit(nullptr));
  setEnv("OFFLOAD_TRACE", "1");

  testing::internal::CaptureStderr();
  ASSERT_SUCCESS(olIteratePlatforms(
      [](ol_platform_handle_t, void *) { return false; }, nullptr));
  ASSERT_EQ(testing::internal::GetCapturedStderr().find("---> "),
            std::string::npos);
  ASSERT_SUCCESS(olShutDown());

  ASSERT_SUCCESS(olInit(nullptr));
  testing::internal::CaptureStderr();
  ASSERT_SUCCESS(olIteratePlatforms(
      [](ol_platform_handle_t, void *) { return false; }, nullptr));
  ASSERT_NE(testing::internal::GetCapturedStderr().find("---> "),
            std::string::npos);
  ASSERT_SUCCESS(olShutDown());
  unsetEnv("OFFLOAD_TRACE");
}

// OlInit is thread safe, concurrent calls are serialized and each one
// counts as exactly one reference.
TEST_F(olInitTest, ConcurrentInit) {
  constexpr size_t NumThreads = 16;
  std::vector<std::thread> Threads;
  std::vector<ol_result_t> Results(NumThreads, nullptr);
  for (size_t I = 0; I < NumThreads; I++)
    Threads.emplace_back([&Results, I] { Results[I] = olInit(nullptr); });
  for (auto &T : Threads)
    T.join();
  for (auto R : Results)
    ASSERT_SUCCESS(R);

  for (size_t I = 0; I < NumThreads; I++)
    ASSERT_SUCCESS(olShutDown());
  ASSERT_ERROR(OL_ERRC_UNINITIALIZED, olShutDown());
}

// OlShutDown is thread safe and each call releases exactly one
// reference.
TEST_F(olInitTest, ConcurrentShutDown) {
  constexpr size_t NumThreads = 16;
  for (size_t I = 0; I < NumThreads; I++)
    ASSERT_SUCCESS(olInit(nullptr));

  std::vector<std::thread> Threads;
  std::vector<ol_result_t> Results(NumThreads, nullptr);
  for (size_t I = 0; I < NumThreads; I++)
    Threads.emplace_back([&Results, I] { Results[I] = olShutDown(); });
  for (auto &T : Threads)
    T.join();
  for (auto R : Results)
    ASSERT_SUCCESS(R);
  ASSERT_ERROR(OL_ERRC_UNINITIALIZED, olShutDown());
}

// A balancing olShutDown from an exit handler is safe even after the library
// shut itself down at process exit, and olInit cannot be used after that.
// Checked from an exit handler in a child process.
//
// On Windows, atexit handlers registered inside a DLL run when the DLL is
// unloaded, after every handler registered by the executable, so the state
// after the library's exit teardown cannot be observed from here.
#ifndef _WIN32
static void checkAfterExitShutDown() {
  if (olShutDown() != OL_SUCCESS)
    _exit(1);
  ol_result_t Res = olInit(nullptr);
  if (!Res || Res->Code != OL_ERRC_UNINITIALIZED)
    _exit(2);
}

TEST_F(olInitTest, ShutDownAtProcessExit) {
  // Re-execute the binary for the child so that no platform, and therefore no
  // library exit handler, exists before the statement runs.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        // Registered first so it runs after the library's own exit handler.
        std::atexit(checkAfterExitShutDown);
        if (olInit(nullptr) != OL_SUCCESS)
          _exit(3);
        // Using a platform registers the library's exit handler.
        if (olIterateDevices([](ol_device_handle_t, void *) { return false; },
                             nullptr) != OL_SUCCESS)
          _exit(4);
        exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}
#endif
