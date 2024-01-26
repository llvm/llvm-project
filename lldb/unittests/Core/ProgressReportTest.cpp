#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Progress.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/Listener.h"
#include "gtest/gtest.h"
#include <thread>

using namespace lldb;
using namespace lldb_private;

namespace {
class ProgressReportTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    PlatformMacOSX::Initialize();
    Debugger::Initialize(nullptr);
  }
  void TearDown() override {
    Debugger::Terminate();
    PlatformMacOSX::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};
} // namespace
TEST_F(ProgressReportTest, TestReportCreation) {
  std::chrono::milliseconds timeout(100);

  // Set up the debugger, make sure that was done properly
  ArchSpec arch("x86_64-apple-macosx-");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  // Get the debugger's broadcaster
  Broadcaster &broadcaster = debugger_sp->GetBroadcaster();

  // Create a listener, make sure it can receive events and that it's
  // listening to the correct broadcast bit
  ListenerSP listener_sp = Listener::MakeListener("progress-listener");

  listener_sp->StartListeningForEvents(&broadcaster,
                                       Debugger::eBroadcastBitProgress);
  EXPECT_TRUE(
      broadcaster.EventTypeHasListeners(Debugger::eBroadcastBitProgress));

  EventSP event_sp;
  const ProgressEventData *data;

  // Scope this for RAII on the progress objects
  // Create progress reports and check that their respective events for having
  // started are broadcasted
  {
    Progress progress1("Progress report 1", "Starting report 1");
    EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));

    data = ProgressEventData::GetEventDataFromEvent(event_sp.get());
    ASSERT_EQ(data->GetDetails(), "Starting report 1");

    Progress progress2("Progress report 2", "Starting report 2");
    EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));

    data = ProgressEventData::GetEventDataFromEvent(event_sp.get());
    ASSERT_EQ(data->GetDetails(), "Starting report 2");

    Progress progress3("Progress report 3", "Starting report 3");
    EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));
    ASSERT_TRUE(event_sp);

    data = ProgressEventData::GetEventDataFromEvent(event_sp.get());
    ASSERT_EQ(data->GetDetails(), "Starting report 3");

    std::this_thread::sleep_for(timeout);
  }

  // Progress report objects should be destroyed at this point so
  // get each report from the queue and check that they've been
  // destroyed in reverse order
  std::this_thread::sleep_for(timeout);
  EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  ASSERT_EQ(data->GetTitle(), "Progress report 3");
  ASSERT_TRUE(data->GetCompleted());

  std::this_thread::sleep_for(timeout);
  EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  ASSERT_EQ(data->GetTitle(), "Progress report 2");
  ASSERT_TRUE(data->GetCompleted());

  std::this_thread::sleep_for(timeout);
  EXPECT_TRUE(listener_sp->GetEvent(event_sp, timeout));
  data = ProgressEventData::GetEventDataFromEvent(event_sp.get());

  ASSERT_EQ(data->GetTitle(), "Progress report 1");
  ASSERT_TRUE(data->GetCompleted());
}
