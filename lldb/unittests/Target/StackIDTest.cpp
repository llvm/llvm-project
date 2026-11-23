//===-- StackIDTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/StackID.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

namespace {
class DummyProcess : public Process {
public:
  DummyProcess(TargetSP target_sp, ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}
  bool CanDebug(TargetSP, bool) override { return true; }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  size_t DoReadMemory(const ProcessAddress &, void *, size_t,
                      Status &) override {
    return 0;
  }
  bool DoUpdateThreadList(ThreadList &, ThreadList &) override { return false; }
  llvm::StringRef GetPluginName() override { return "Dummy"; }
  CFAComparison CompareCFAs(const StackID &, const StackID &) override {
    return m_opinion;
  }

  CFAComparison m_opinion = CFAComparison::NoOpinion;
};

class StackIDTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, platform_linux::PlatformLinux> subsystems;

public:
  void SetUp() override {
    ArchSpec arch("x86_64-pc-linux");
    Platform::SetHostPlatform(
        platform_linux::PlatformLinux::CreateInstance(true, &arch));
    m_debugger_sp = Debugger::CreateInstance();
    ASSERT_TRUE(m_debugger_sp);
    TargetSP target_sp;
    PlatformSP platform_sp;
    ASSERT_TRUE(m_debugger_sp->GetTargetList()
                    .CreateTarget(*m_debugger_sp, "", arch, eLoadDependentsNo,
                                  platform_sp, target_sp)
                    .Success());
    m_process_sp = std::make_shared<DummyProcess>(
        target_sp, Listener::MakeListener("dummy"));
  }

protected:
  DebuggerSP m_debugger_sp;
  std::shared_ptr<DummyProcess> m_process_sp;
};
} // namespace

TEST_F(StackIDTest, TailCallDepthOrdersFramesSharingACFA) {
  Process &process = *m_process_sp;
  StackID unwound(0x1000, 0x7f00, nullptr, nullptr);
  StackID tail_caller = unwound;
  tail_caller.SetTailCallDepth(1);
  StackID outer_tail_caller = unwound;
  outer_tail_caller.SetTailCallDepth(2);
  StackID caller(0x1000, 0x7f10, nullptr, nullptr);

  EXPECT_NE(unwound, tail_caller);
  EXPECT_NE(tail_caller, outer_tail_caller);
  EXPECT_TRUE(unwound.IsYoungerThan(tail_caller, process));
  EXPECT_FALSE(tail_caller.IsYoungerThan(unwound, process));
  EXPECT_TRUE(tail_caller.IsYoungerThan(outer_tail_caller, process));
  EXPECT_FALSE(outer_tail_caller.IsYoungerThan(tail_caller, process));
  EXPECT_TRUE(outer_tail_caller.IsYoungerThan(caller, process));
  EXPECT_FALSE(caller.IsYoungerThan(outer_tail_caller, process));
}

TEST_F(StackIDTest, ProcessOrdersCFAs) {
  StackID younger(0x1000, 0x7f00, nullptr, nullptr);
  StackID older(0x1000, 0x7f10, nullptr, nullptr);
  // Without an opinion from the process, the stack is assumed to grow down.
  EXPECT_TRUE(younger.IsYoungerThan(older, *m_process_sp));

  m_process_sp->m_opinion = CFAComparison::Older;
  EXPECT_FALSE(younger.IsYoungerThan(older, *m_process_sp));
}
