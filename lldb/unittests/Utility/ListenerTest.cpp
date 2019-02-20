//===-- ListenerTest.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Utility/Broadcaster.h"
#include "lldb/Utility/Listener.h"
#include <future>
#include <thread>

using namespace lldb;
using namespace lldb_private;

TEST(ListenerTest, GetEventImmediate) {
  EventSP event_sp;
  Broadcaster broadcaster(nullptr, "test-broadcaster");

  // Create a listener, sign it up, make sure it receives an event.
  ListenerSP listener_sp = Listener::MakeListener("test-listener");
  const uint32_t event_mask = 1;
  ASSERT_EQ(event_mask,
            listener_sp->StartListeningForEvents(&broadcaster, event_mask));

  const std::chrono::seconds timeout(0);
  // Without any events sent, these should return false.
  EXPECT_FALSE(listener_sp->GetEvent(event_sp, timeout));
  EXPECT_FALSE(listener_sp->GetEventForBroadcaster(nullptr, event_sp, timeout));
  EXPECT_FALSE(
      listener_sp->GetEventForBroadcaster(&broadcaster, event_sp, timeout));
  EXPECT_FALSE(listener_sp->GetEventForBroadcasterWithType(
      &broadcaster, event_mask, event_sp, timeout));

  // Now send events and make sure they get it.
  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));

  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_TRUE(listener_sp->GetEventForBroadcaster(nullptr, event_sp, timeout));

  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_TRUE(
      listener_sp->GetEventForBroadcaster(&broadcaster, event_sp, timeout));

  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_FALSE(listener_sp->GetEventForBroadcasterWithType(
      &broadcaster, event_mask * 2, event_sp, timeout));
  EXPECT_TRUE(listener_sp->GetEventForBroadcasterWithType(
      &broadcaster, event_mask, event_sp, timeout));
}

TEST(ListenerTest, GetEventWait) {
  EventSP event_sp;
  Broadcaster broadcaster(nullptr, "test-broadcaster");

  // Create a listener, sign it up, make sure it receives an event.
  ListenerSP listener_sp = Listener::MakeListener("test-listener");
  const uint32_t event_mask = 1;
  ASSERT_EQ(event_mask,
            listener_sp->StartListeningForEvents(&broadcaster, event_mask));

  // Without any events sent, these should make a short wait and return false.
  std::chrono::microseconds timeout(10);
  EXPECT_FALSE(listener_sp->GetEvent(event_sp, timeout));
  EXPECT_FALSE(listener_sp->GetEventForBroadcaster(nullptr, event_sp, timeout));
  EXPECT_FALSE(
      listener_sp->GetEventForBroadcaster(&broadcaster, event_sp, timeout));
  EXPECT_FALSE(listener_sp->GetEventForBroadcasterWithType(
      &broadcaster, event_mask, event_sp, timeout));

  // Now send events and make sure they get it.
  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));

  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_TRUE(listener_sp->GetEventForBroadcaster(nullptr, event_sp, timeout));

  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_TRUE(
      listener_sp->GetEventForBroadcaster(&broadcaster, event_sp, timeout));

  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_FALSE(listener_sp->GetEventForBroadcasterWithType(
      &broadcaster, event_mask * 2, event_sp, timeout));
  EXPECT_TRUE(listener_sp->GetEventForBroadcasterWithType(
      &broadcaster, event_mask, event_sp, timeout));

  auto delayed_broadcast = [&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    broadcaster.BroadcastEvent(event_mask, nullptr);
  };

  // These should do an infinite wait at return the event our asynchronous
  // broadcast sends.
  std::future<void> async_broadcast =
      std::async(std::launch::async, delayed_broadcast);
  EXPECT_TRUE(listener_sp->GetEvent(event_sp, llvm::None));
  async_broadcast.get();

  async_broadcast = std::async(std::launch::async, delayed_broadcast);
  EXPECT_TRUE(
      listener_sp->GetEventForBroadcaster(&broadcaster, event_sp, llvm::None));
  async_broadcast.get();

  async_broadcast = std::async(std::launch::async, delayed_broadcast);
  EXPECT_TRUE(listener_sp->GetEventForBroadcasterWithType(
      &broadcaster, event_mask, event_sp, llvm::None));
  async_broadcast.get();
}
