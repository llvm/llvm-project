//===-- DiagnosticEventTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/DebuggerEvents.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/Broadcaster.h"
#include "lldb/Utility/Event.h"
#include "lldb/Utility/Listener.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::repro;

static const constexpr std::chrono::seconds TIMEOUT(0);
static const constexpr size_t DEBUGGERS = 3;

static std::once_flag debugger_initialize_flag;

namespace {
class DiagnosticEventTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    PlatformMacOSX::Initialize();
    std::call_once(debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
    ArchSpec arch("x86_64-apple-macosx-");
    Platform::SetHostPlatform(
        PlatformRemoteMacOSX::CreateInstance(true, &arch, /*debugger=*/nullptr,
                                             /*metadata=*/nullptr));
  }
  void TearDown() override {
    PlatformMacOSX::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};
} // namespace

TEST_F(DiagnosticEventTest, Warning) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();

  Broadcaster &broadcaster = debugger_sp->GetBroadcaster();
  ListenerSP listener_sp = Listener::MakeListener("test-listener");

  listener_sp->StartListeningForEvents(&broadcaster,
                                       Debugger::eBroadcastBitWarning);
  EXPECT_TRUE(
      broadcaster.EventTypeHasListeners(Debugger::eBroadcastBitWarning));

  Debugger::ReportWarning("foo", debugger_sp->GetID());

  EventSP event_sp;
  EXPECT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  ASSERT_TRUE(event_sp);

  const DiagnosticEventData *data =
      DiagnosticEventData::GetEventDataFromEvent(event_sp.get());
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(data->GetPrefix(), "warning");
  EXPECT_EQ(data->GetMessage(), "foo");

  Debugger::Destroy(debugger_sp);
}

TEST_F(DiagnosticEventTest, Error) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();

  Broadcaster &broadcaster = debugger_sp->GetBroadcaster();
  ListenerSP listener_sp = Listener::MakeListener("test-listener");

  listener_sp->StartListeningForEvents(&broadcaster,
                                       Debugger::eBroadcastBitError);
  EXPECT_TRUE(broadcaster.EventTypeHasListeners(Debugger::eBroadcastBitError));

  Debugger::ReportError("bar", debugger_sp->GetID());

  EventSP event_sp;
  EXPECT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  ASSERT_TRUE(event_sp);

  const DiagnosticEventData *data =
      DiagnosticEventData::GetEventDataFromEvent(event_sp.get());
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(data->GetPrefix(), "error");
  EXPECT_EQ(data->GetMessage(), "bar");

  Debugger::Destroy(debugger_sp);
}

TEST_F(DiagnosticEventTest, MultipleDebuggers) {
  std::vector<DebuggerSP> debuggers;
  std::vector<ListenerSP> listeners;

  for (size_t i = 0; i < DEBUGGERS; ++i) {
    DebuggerSP debugger = Debugger::CreateInstance();
    ListenerSP listener = Listener::MakeListener("listener");

    debuggers.push_back(debugger);
    listeners.push_back(listener);

    listener->StartListeningForEvents(&debugger->GetBroadcaster(),
                                      Debugger::eBroadcastBitError);
  }

  Debugger::ReportError("baz");

  for (size_t i = 0; i < DEBUGGERS; ++i) {
    EventSP event_sp;
    EXPECT_TRUE(listeners[i]->GetEvent(event_sp, TIMEOUT));
    ASSERT_TRUE(event_sp);

    const DiagnosticEventData *data =
        DiagnosticEventData::GetEventDataFromEvent(event_sp.get());
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->GetPrefix(), "error");
    EXPECT_EQ(data->GetMessage(), "baz");
  }

  for (size_t i = 0; i < DEBUGGERS; ++i) {
    Debugger::Destroy(debuggers[i]);
  }
}

TEST_F(DiagnosticEventTest, WarningOnce) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();

  Broadcaster &broadcaster = debugger_sp->GetBroadcaster();
  ListenerSP listener_sp = Listener::MakeListener("test-listener");

  listener_sp->StartListeningForEvents(&broadcaster,
                                       Debugger::eBroadcastBitWarning);
  EXPECT_TRUE(
      broadcaster.EventTypeHasListeners(Debugger::eBroadcastBitWarning));

  std::once_flag once;
  Debugger::ReportWarning("foo", debugger_sp->GetID(), &once);

  {
    EventSP event_sp;
    EXPECT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
    ASSERT_TRUE(event_sp);

    const DiagnosticEventData *data =
        DiagnosticEventData::GetEventDataFromEvent(event_sp.get());
    ASSERT_NE(data, nullptr);
    EXPECT_EQ(data->GetPrefix(), "warning");
    EXPECT_EQ(data->GetMessage(), "foo");
  }

  EventSP second_event_sp;
  EXPECT_FALSE(listener_sp->GetEvent(second_event_sp, TIMEOUT));

  Debugger::ReportWarning("foo", debugger_sp->GetID(), &once);
  EXPECT_FALSE(listener_sp->GetEvent(second_event_sp, TIMEOUT));

  Debugger::ReportWarning("foo", debugger_sp->GetID());
  EXPECT_TRUE(listener_sp->GetEvent(second_event_sp, TIMEOUT));

  Debugger::Destroy(debugger_sp);
}
