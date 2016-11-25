//===-- ListenerTest.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Listener.h"
#include <future>
#include <thread>

using namespace lldb;
using namespace lldb_private;

TEST(ListenerTest, GetNextEvent) {
  EventSP event_sp;
  Broadcaster broadcaster(nullptr, "test-broadcaster");

  // Create a listener, sign it up, make sure it recieves an event.
  ListenerSP listener_sp = Listener::MakeListener("test-listener");
  const uint32_t event_mask = 1;
  ASSERT_EQ(event_mask,
            listener_sp->StartListeningForEvents(&broadcaster, event_mask));

  // Without any events sent, these should return false.
  EXPECT_FALSE(listener_sp->GetNextEvent(event_sp));
  EXPECT_FALSE(listener_sp->GetNextEventForBroadcaster(nullptr, event_sp));
  EXPECT_FALSE(listener_sp->GetNextEventForBroadcaster(&broadcaster, event_sp));
  EXPECT_FALSE(listener_sp->GetNextEventForBroadcasterWithType(
      &broadcaster, event_mask, event_sp));

  // Now send events and make sure they get it.
  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_TRUE(listener_sp->GetNextEvent(event_sp));

  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_TRUE(listener_sp->GetNextEventForBroadcaster(nullptr, event_sp));

  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_TRUE(listener_sp->GetNextEventForBroadcaster(&broadcaster, event_sp));

  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_FALSE(listener_sp->GetNextEventForBroadcasterWithType(
      &broadcaster, event_mask * 2, event_sp));
  EXPECT_TRUE(listener_sp->GetNextEventForBroadcasterWithType(
      &broadcaster, event_mask, event_sp));
}

TEST(ListenerTest, WaitForEvent) {
  EventSP event_sp;
  Broadcaster broadcaster(nullptr, "test-broadcaster");

  // Create a listener, sign it up, make sure it recieves an event.
  ListenerSP listener_sp = Listener::MakeListener("test-listener");
  const uint32_t event_mask = 1;
  ASSERT_EQ(event_mask,
            listener_sp->StartListeningForEvents(&broadcaster, event_mask));

  // Without any events sent, these should make a short wait and return false.
  std::chrono::microseconds timeout(10);
  EXPECT_FALSE(listener_sp->WaitForEvent(timeout, event_sp));
  EXPECT_FALSE(
      listener_sp->WaitForEventForBroadcaster(timeout, nullptr, event_sp));
  EXPECT_FALSE(
      listener_sp->WaitForEventForBroadcaster(timeout, &broadcaster, event_sp));
  EXPECT_FALSE(listener_sp->WaitForEventForBroadcasterWithType(
      timeout, &broadcaster, event_mask, event_sp));

  // Now send events and make sure they get it.
  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_TRUE(listener_sp->WaitForEvent(timeout, event_sp));

  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_TRUE(
      listener_sp->WaitForEventForBroadcaster(timeout, nullptr, event_sp));

  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_TRUE(
      listener_sp->WaitForEventForBroadcaster(timeout, &broadcaster, event_sp));

  broadcaster.BroadcastEvent(event_mask, nullptr);
  EXPECT_FALSE(listener_sp->WaitForEventForBroadcasterWithType(
      timeout, &broadcaster, event_mask * 2, event_sp));
  EXPECT_TRUE(listener_sp->WaitForEventForBroadcasterWithType(
      timeout, &broadcaster, event_mask, event_sp));

  timeout = std::chrono::seconds(0);
  auto delayed_broadcast = [&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    broadcaster.BroadcastEvent(event_mask, nullptr);
  };

  // These should do an infinite wait at return the event our asynchronous
  // broadcast sends.
  std::future<void> async_broadcast =
      std::async(std::launch::async, delayed_broadcast);
  EXPECT_TRUE(listener_sp->WaitForEvent(timeout, event_sp));
  async_broadcast.get();

  async_broadcast = std::async(std::launch::async, delayed_broadcast);
  EXPECT_TRUE(
      listener_sp->WaitForEventForBroadcaster(timeout, &broadcaster, event_sp));
  async_broadcast.get();

  async_broadcast = std::async(std::launch::async, delayed_broadcast);
  EXPECT_TRUE(listener_sp->WaitForEventForBroadcasterWithType(
      timeout, &broadcaster, event_mask, event_sp));
  async_broadcast.get();
}
