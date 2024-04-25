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
#include <memory>
#include <mutex>

using namespace lldb;
using namespace lldb_private;

static std::chrono::milliseconds TIMEOUT(500);

class ProgressReportTest : public ::testing::Test {
public:
  ListenerSP CreateListenerFor(uint32_t bit) {
    // Set up the debugger, make sure that was done properly.
    ArchSpec arch("x86_64-apple-macosx-");
    Platform::SetHostPlatform(
        PlatformRemoteMacOSX::CreateInstance(true, &arch));

    m_debugger_sp = Debugger::CreateInstance();

    // Get the debugger's broadcaster.
    Broadcaster &broadcaster = m_debugger_sp->GetBroadcaster();

    // Create a listener, make sure it can receive events and that it's
    // listening to the correct broadcast bit.
    m_listener_sp = Listener::MakeListener("progress-listener");
    m_listener_sp->StartListeningForEvents(&broadcaster, bit);
    return m_listener_sp;
  }

protected:
  // The debugger's initialization function can't be called with no arguments
  // so calling it using SubsystemRAII will cause the test build to fail as
  // SubsystemRAII will call Initialize with no arguments. As such we set it up
  // here the usual way.
  void SetUp() override {
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
  };

  DebuggerSP m_debugger_sp;
  ListenerSP m_listener_sp;
  SubsystemRAII<FileSystem, HostInfo, PlatformMacOSX, ProgressManager>
      subsystems;
};

TEST_F(ProgressReportTest, TestReportCreation) {
  ListenerSP listener_sp = CreateListenerFor(Debugger::eBroadcastBitProgress);
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
  ASSERT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  EXPECT_EQ(data->GetDetails(), "Starting report 1");
  EXPECT_FALSE(data->IsFinite());
  EXPECT_FALSE(data->GetCompleted());
  EXPECT_EQ(data->GetTotal(), Progress::kNonDeterministicTotal);
  EXPECT_EQ(data->GetMessage(), "Progress report 1: Starting report 1");

  ASSERT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  EXPECT_EQ(data->GetDetails(), "Starting report 2");
  EXPECT_FALSE(data->IsFinite());
  EXPECT_FALSE(data->GetCompleted());
  EXPECT_EQ(data->GetTotal(), Progress::kNonDeterministicTotal);
  EXPECT_EQ(data->GetMessage(), "Progress report 2: Starting report 2");

  ASSERT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  EXPECT_EQ(data->GetDetails(), "Starting report 3");
  EXPECT_FALSE(data->IsFinite());
  EXPECT_FALSE(data->GetCompleted());
  EXPECT_EQ(data->GetTotal(), Progress::kNonDeterministicTotal);
  EXPECT_EQ(data->GetMessage(), "Progress report 3: Starting report 3");

  // Progress report objects should be destroyed at this point so
  // get each report from the queue and check that they've been
  // destroyed in reverse order.
  ASSERT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  EXPECT_EQ(data->GetTitle(), "Progress report 3");
  EXPECT_TRUE(data->GetCompleted());
  EXPECT_FALSE(data->IsFinite());
  EXPECT_EQ(data->GetMessage(), "Progress report 3: Starting report 3");

  ASSERT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  EXPECT_EQ(data->GetTitle(), "Progress report 2");
  EXPECT_TRUE(data->GetCompleted());
  EXPECT_FALSE(data->IsFinite());
  EXPECT_EQ(data->GetMessage(), "Progress report 2: Starting report 2");

  ASSERT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  EXPECT_EQ(data->GetTitle(), "Progress report 1");
  EXPECT_TRUE(data->GetCompleted());
  EXPECT_FALSE(data->IsFinite());
  EXPECT_EQ(data->GetMessage(), "Progress report 1: Starting report 1");
}

TEST_F(ProgressReportTest, TestProgressManager) {
  ListenerSP listener_sp =
      CreateListenerFor(Debugger::eBroadcastBitProgressCategory);
  EventSP event_sp;
  const ProgressEventData *data;

  // Create three progress events with the same category then try to pop 2
  // events from the queue in a row before the progress reports are destroyed.
  // Since only 1 event should've been broadcast for this category, the second
  // GetEvent() call should return false.
  {
    Progress progress1("Progress report 1", "Starting report 1");
    Progress progress2("Progress report 1", "Starting report 2");
    Progress progress3("Progress report 1", "Starting report 3");
    ASSERT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
    ASSERT_FALSE(listener_sp->GetEvent(event_sp, TIMEOUT));
  }

  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  EXPECT_EQ(data->GetDetails(), "");
  EXPECT_FALSE(data->IsFinite());
  EXPECT_FALSE(data->GetCompleted());
  EXPECT_EQ(data->GetTotal(), Progress::kNonDeterministicTotal);
  EXPECT_EQ(data->GetMessage(), "Progress report 1");

  // Pop another event from the queue, this should be the event for the final
  // report for this category.
  ASSERT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  EXPECT_EQ(data->GetDetails(), "");
  EXPECT_FALSE(data->IsFinite());
  EXPECT_TRUE(data->GetCompleted());
  EXPECT_EQ(data->GetTotal(), Progress::kNonDeterministicTotal);
  EXPECT_EQ(data->GetMessage(), "Progress report 1");
}

TEST_F(ProgressReportTest, TestOverlappingEvents) {
  ListenerSP listener_sp =
      CreateListenerFor(Debugger::eBroadcastBitProgressCategory);
  EventSP event_sp;
  const ProgressEventData *data;

  // Create two progress reports of the same category that overlap with each
  // other. Here we want to ensure that the ID broadcasted for the initial and
  // final reports for this category are the same.
  std::unique_ptr<Progress> overlap_progress1 =
      std::make_unique<Progress>("Overlapping report 1", "Starting report 1");
  std::unique_ptr<Progress> overlap_progress2 =
      std::make_unique<Progress>("Overlapping report 1", "Starting report 2");
  overlap_progress1.reset();

  ASSERT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());
  // Get the ID used in the first report for this category.
  uint64_t expected_progress_id = data->GetID();

  EXPECT_EQ(data->GetDetails(), "");
  EXPECT_FALSE(data->IsFinite());
  EXPECT_FALSE(data->GetCompleted());
  EXPECT_EQ(data->GetTotal(), Progress::kNonDeterministicTotal);
  EXPECT_EQ(data->GetMessage(), "Overlapping report 1");

  overlap_progress2.reset();

  ASSERT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  EXPECT_EQ(data->GetDetails(), "");
  EXPECT_FALSE(data->IsFinite());
  EXPECT_TRUE(data->GetCompleted());
  EXPECT_EQ(data->GetTotal(), Progress::kNonDeterministicTotal);
  EXPECT_EQ(data->GetMessage(), "Overlapping report 1");
  // The progress ID for the final report should be the same as that for the
  // initial report.
  EXPECT_EQ(data->GetID(), expected_progress_id);
}

TEST_F(ProgressReportTest, TestProgressManagerDisjointReports) {
  ListenerSP listener_sp =
      CreateListenerFor(Debugger::eBroadcastBitProgressCategory);
  EventSP event_sp;
  const ProgressEventData *data;
  uint64_t expected_progress_id;

  { Progress progress("Coalesced report 1", "Starting report 1"); }
  { Progress progress("Coalesced report 1", "Starting report 2"); }
  { Progress progress("Coalesced report 1", "Starting report 3"); }

  ASSERT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());
  expected_progress_id = data->GetID();

  EXPECT_EQ(data->GetDetails(), "");
  EXPECT_FALSE(data->IsFinite());
  EXPECT_FALSE(data->GetCompleted());
  EXPECT_EQ(data->GetTotal(), Progress::kNonDeterministicTotal);
  EXPECT_EQ(data->GetMessage(), "Coalesced report 1");

  ASSERT_TRUE(listener_sp->GetEvent(event_sp, TIMEOUT));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  EXPECT_EQ(data->GetID(), expected_progress_id);
  EXPECT_EQ(data->GetDetails(), "");
  EXPECT_FALSE(data->IsFinite());
  EXPECT_TRUE(data->GetCompleted());
  EXPECT_EQ(data->GetTotal(), Progress::kNonDeterministicTotal);
  EXPECT_EQ(data->GetMessage(), "Coalesced report 1");

  ASSERT_FALSE(listener_sp->GetEvent(event_sp, TIMEOUT));
}
