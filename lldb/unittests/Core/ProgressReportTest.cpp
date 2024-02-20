//===-- ProgressReportTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Progress.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/Listener.h"
#include "gtest/gtest.h"
#include <mutex>
#include <thread>

using namespace lldb;
using namespace lldb_private;

class ProgressReportTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, PlatformMacOSX> subsystems;

  // The debugger's initialization function can't be called with no arguments
  // so calling it using SubsystemRAII will cause the test build to fail as
  // SubsystemRAII will call Initialize with no arguments. As such we set it up
  // here the usual way.
  void SetUp() override {
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
  };
};

TEST_F(ProgressReportTest, TestReportCreation) {
  std::chrono::milliseconds timeout(100);

  // Set up the debugger, make sure that was done properly.
  ArchSpec arch("x86_64-apple-macosx-");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  // Get the debugger's broadcaster.
  Broadcaster &broadcaster = debugger_sp->GetBroadcaster();

  // Create a listener, make sure it can receive events and that it's
  // listening to the correct broadcast bit.
  ListenerSP listener_sp = Listener::MakeListener("progress-listener");

  listener_sp->StartListeningForEvents(&broadcaster,
                                       Debugger::eBroadcastBitProgress);
  EXPECT_TRUE(
      broadcaster.EventTypeHasListeners(Debugger::eBroadcastBitProgress));

  EventSP event_sp;
  const ProgressEventData *data;

  // Scope this for RAII on the progress objects.
  // Create progress reports and check that their respective events for having
  // started and ended are broadcasted.
  {
    Progress progress1("Progress report 1", "Starting report 1");
    Progress progress2("Progress report 2", "Starting report 2");
    Progress progress3("Progress report 3", "Starting report 3");
  }

  // Start popping events from the queue, they should have been recevied
  // in this order:
  // Starting progress: 1, 2, 3
  // Ending progress: 3, 2, 1
  EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  ASSERT_EQ(data->GetDetails(), "Starting report 1");
  ASSERT_FALSE(data->IsFinite());
  ASSERT_FALSE(data->GetCompleted());
  ASSERT_EQ(data->GetTotal(), Progress::kNonDeterministicTotal);
  ASSERT_EQ(data->GetMessage(), "Progress report 1: Starting report 1");

  EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  ASSERT_EQ(data->GetDetails(), "Starting report 2");
  ASSERT_FALSE(data->IsFinite());
  ASSERT_FALSE(data->GetCompleted());
  ASSERT_EQ(data->GetTotal(), Progress::kNonDeterministicTotal);
  ASSERT_EQ(data->GetMessage(), "Progress report 2: Starting report 2");

  EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());
  ASSERT_EQ(data->GetDetails(), "Starting report 3");
  ASSERT_FALSE(data->IsFinite());
  ASSERT_FALSE(data->GetCompleted());
  ASSERT_EQ(data->GetTotal(), Progress::kNonDeterministicTotal);
  ASSERT_EQ(data->GetMessage(), "Progress report 3: Starting report 3");

  // Progress report objects should be destroyed at this point so
  // get each report from the queue and check that they've been
  // destroyed in reverse order.
  EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  ASSERT_EQ(data->GetTitle(), "Progress report 3");
  ASSERT_TRUE(data->GetCompleted());
  ASSERT_FALSE(data->IsFinite());
  ASSERT_EQ(data->GetMessage(), "Progress report 3: Starting report 3");

  EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  ASSERT_EQ(data->GetTitle(), "Progress report 2");
  ASSERT_TRUE(data->GetCompleted());
  ASSERT_FALSE(data->IsFinite());
  ASSERT_EQ(data->GetMessage(), "Progress report 2: Starting report 2");

  EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  ASSERT_EQ(data->GetTitle(), "Progress report 1");
  ASSERT_TRUE(data->GetCompleted());
  ASSERT_FALSE(data->IsFinite());
  ASSERT_EQ(data->GetMessage(), "Progress report 1: Starting report 1");
}
