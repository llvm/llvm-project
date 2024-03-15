//===-- AlarmTest.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Alarm.h"
#include "gtest/gtest.h"

#include <chrono>
#include <thread>

using namespace lldb_private;
using namespace std::chrono_literals;

// Increase the timeout tenfold when running under ASan as it can have about the
// same performance overhead.
#if __has_feature(address_sanitizer)
static constexpr auto TEST_TIMEOUT = 10000ms;
#else
static constexpr auto TEST_TIMEOUT = 1000ms;
#endif

// The time between scheduling a callback and it getting executed. This should
// NOT be increased under ASan.
static constexpr auto ALARM_TIMEOUT = 500ms;

// If there are any pending callbacks, make sure they run before the Alarm
// object is destroyed.
static constexpr bool RUN_CALLBACKS_ON_EXIT = true;

TEST(AlarmTest, Create) {
  std::mutex m;

  std::vector<Alarm::TimePoint> callbacks_actual;
  std::vector<Alarm::TimePoint> callbacks_expected;

  Alarm alarm(ALARM_TIMEOUT, RUN_CALLBACKS_ON_EXIT);

  // Create 5 alarms some time apart.
  for (size_t i = 0; i < 5; ++i) {
    callbacks_actual.emplace_back();
    callbacks_expected.emplace_back(std::chrono::system_clock::now() +
                                    ALARM_TIMEOUT);

    alarm.Create([&callbacks_actual, &m, i]() {
      std::lock_guard<std::mutex> guard(m);
      callbacks_actual[i] = std::chrono::system_clock::now();
    });

    std::this_thread::sleep_for(ALARM_TIMEOUT / 5);
  }

  // Leave plenty of time for all the alarms to fire.
  std::this_thread::sleep_for(TEST_TIMEOUT);

  // Make sure all the alarms fired around the expected time.
  for (size_t i = 0; i < 5; ++i)
    EXPECT_GE(callbacks_actual[i], callbacks_expected[i]);
}

TEST(AlarmTest, Exit) {
  std::mutex m;

  std::vector<Alarm::Handle> handles;
  std::vector<bool> callbacks;

  {
    Alarm alarm(ALARM_TIMEOUT, RUN_CALLBACKS_ON_EXIT);

    // Create 5 alarms.
    for (size_t i = 0; i < 5; ++i) {
      callbacks.emplace_back(false);

      handles.push_back(alarm.Create([&callbacks, &m, i]() {
        std::lock_guard<std::mutex> guard(m);
        callbacks[i] = true;
      }));
    }

    // Let the alarm go out of scope before any alarm had a chance to fire.
  }

  // Make sure none of the alarms fired.
  for (bool callback : callbacks)
    EXPECT_TRUE(callback);
}

TEST(AlarmTest, Cancel) {
  std::mutex m;

  std::vector<Alarm::Handle> handles;
  std::vector<bool> callbacks;

  Alarm alarm(ALARM_TIMEOUT, RUN_CALLBACKS_ON_EXIT);

  // Create 5 alarms.
  for (size_t i = 0; i < 5; ++i) {
    callbacks.emplace_back(false);

    handles.push_back(alarm.Create([&callbacks, &m, i]() {
      std::lock_guard<std::mutex> guard(m);
      callbacks[i] = true;
    }));
  }

  // Make sure we can cancel the first 4 alarms.
  for (size_t i = 0; i < 4; ++i)
    EXPECT_TRUE(alarm.Cancel(handles[i]));

  // Leave plenty of time for all the alarms to fire.
  std::this_thread::sleep_for(TEST_TIMEOUT);

  // Make sure none of the first 4 alarms fired.
  for (size_t i = 0; i < 4; ++i)
    EXPECT_FALSE(callbacks[i]);

  // Make sure the fifth alarm still fired.
  EXPECT_TRUE(callbacks[4]);
}

TEST(AlarmTest, Restart) {
  std::mutex m;

  std::vector<Alarm::Handle> handles;
  std::vector<Alarm::TimePoint> callbacks_actual;
  std::vector<Alarm::TimePoint> callbacks_expected;

  Alarm alarm(ALARM_TIMEOUT, RUN_CALLBACKS_ON_EXIT);

  // Create 5 alarms some time apart.
  for (size_t i = 0; i < 5; ++i) {
    callbacks_actual.emplace_back();
    callbacks_expected.emplace_back(std::chrono::system_clock::now() +
                                    ALARM_TIMEOUT);

    handles.push_back(alarm.Create([&callbacks_actual, &m, i]() {
      std::lock_guard<std::mutex> guard(m);
      callbacks_actual[i] = std::chrono::system_clock::now();
    }));

    std::this_thread::sleep_for(ALARM_TIMEOUT / 5);
  }

  // Update the last 2 alarms.
  for (size_t i = 3; i < 5; ++i) {
    callbacks_expected[i] = std::chrono::system_clock::now() + ALARM_TIMEOUT;
    EXPECT_TRUE(alarm.Restart(handles[i]));
  }

  // Leave plenty of time for all the alarms to fire.
  std::this_thread::sleep_for(TEST_TIMEOUT);

  // Make sure all the alarms around the expected time.
  for (size_t i = 0; i < 5; ++i)
    EXPECT_GE(callbacks_actual[i], callbacks_expected[i]);
}
