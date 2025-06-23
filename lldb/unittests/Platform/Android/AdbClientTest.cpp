//===-- AdbClientTest.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/Android/AdbClient.h"
#include "gtest/gtest.h"
#include <atomic>
#include <cstdlib>
#include <future>
#include <thread>
#include <vector>

static void set_env(const char *var, const char *value) {
#ifdef _WIN32
  _putenv_s(var, value);
#else
  setenv(var, value, true);
#endif
}

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {
namespace platform_android {

class AdbClientTest : public ::testing::Test {
public:
  void SetUp() override { set_env("ANDROID_SERIAL", ""); }

  void TearDown() override { set_env("ANDROID_SERIAL", ""); }
};

TEST_F(AdbClientTest, CreateByDeviceId) {
  AdbClient adb;
  Status error = AdbClient::CreateByDeviceID("device1", adb);
  EXPECT_TRUE(error.Success());
  EXPECT_EQ("device1", adb.GetDeviceID());
}

TEST_F(AdbClientTest, CreateByDeviceId_ByEnvVar) {
  set_env("ANDROID_SERIAL", "device2");

  AdbClient adb;
  Status error = AdbClient::CreateByDeviceID("", adb);
  EXPECT_TRUE(error.Success());
  EXPECT_EQ("device2", adb.GetDeviceID());
}

TEST_F(AdbClientTest, GetSyncServiceThreadSafe) {
  // Test high-volume concurrent access to GetSyncService
  // This test verifies thread safety under sustained load with many rapid calls
  // Catches race conditions that emerge when multiple threads make repeated
  // calls to GetSyncService on the same AdbClient instance

  AdbClient shared_adb_client("test_device");

  const int num_threads = 8;
  std::vector<std::future<bool>> futures;
  std::atomic<int> success_count{0};
  std::atomic<int> null_count{0};

  // Launch multiple threads that all call GetSyncService on the SAME AdbClient
  for (int i = 0; i < num_threads; ++i) {
    futures.push_back(std::async(std::launch::async, [&]() {
      // Multiple rapid calls to trigger the race condition
      for (int j = 0; j < 20; ++j) {
        Status error;

        auto sync_service = shared_adb_client.GetSyncService(error);

        if (sync_service != nullptr) {
          success_count++;
        } else {
          null_count++;
        }

        // Small delay to increase chance of hitting the race condition
        std::this_thread::sleep_for(std::chrono::microseconds(1));
      }
      return true;
    }));
  }

  // Wait for all threads to complete
  bool all_completed = true;
  for (auto &future : futures) {
    bool thread_result = future.get();
    if (!thread_result) {
      all_completed = false;
    }
  }

  // This should pass (though sync services may fail
  // to connect)
  EXPECT_TRUE(all_completed) << "Parallel GetSyncService calls should not "
                                "crash due to race conditions. "
                             << "Successes: " << success_count.load()
                             << ", Nulls: " << null_count.load();

  // The key test: we should complete all operations without crashing
  int total_operations = num_threads * 20;
  int completed_operations = success_count.load() + null_count.load();
  EXPECT_EQ(total_operations, completed_operations)
      << "All operations should complete without crashing";
}

TEST_F(AdbClientTest, ConnectionMoveRaceCondition) {
  // Test simultaneous access timing to GetSyncService
  // This test verifies thread safety when multiple threads start at exactly
  // the same time, maximizing the chance of hitting precise timing conflicts
  // Catches race conditions that occur with synchronized simultaneous access

  AdbClient adb_client("test_device");

  // Try to trigger the exact race condition by having multiple threads
  // simultaneously call GetSyncService

  std::atomic<bool> start_flag{false};
  std::vector<std::thread> threads;
  std::atomic<int> null_service_count{0};
  std::atomic<int> valid_service_count{0};

  const int num_threads = 10;

  // Create threads that will all start simultaneously
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&]() {
      // Wait for all threads to be ready
      while (!start_flag.load()) {
        std::this_thread::yield();
      }

      Status error;
      auto sync_service = adb_client.GetSyncService(error);

      if (sync_service == nullptr) {
        null_service_count++;
      } else {
        valid_service_count++;
      }
    });
  }

  // Start all threads simultaneously to maximize chance of race condition
  start_flag.store(true);

  // Wait for all threads to complete
  for (auto &thread : threads) {
    thread.join();
  }

  // The test passes if we don't crash
  int total_results = null_service_count.load() + valid_service_count.load();
  EXPECT_EQ(num_threads, total_results)
      << "All threads should complete without crashing. "
      << "Null services: " << null_service_count.load()
      << ", Valid services: " << valid_service_count.load();
}

} // end namespace platform_android
} // end namespace lldb_private
