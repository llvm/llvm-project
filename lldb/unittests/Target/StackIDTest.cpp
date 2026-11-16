//===-- StackIDTest.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/StackID.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;

namespace {
/// A process whose memory below 0x1000 is unmapped, whose memory below
/// 0x10000000 is heap memory, and whose other memory is stack memory.
class MemoryRegionProcess : public Process {
public:
  MemoryRegionProcess(TargetSP target_sp, ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}

  bool CanDebug(TargetSP, bool) override { return true; }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  size_t DoReadMemory(const ProcessAddress &, void *, size_t,
                      Status &) override {
    return 0;
  }
  bool DoUpdateThreadList(ThreadList &, ThreadList &) override { return false; }
  llvm::StringRef GetPluginName() override { return "memory-region"; }

protected:
  Status DoGetMemoryRegionInfo(addr_t load_addr,
                               MemoryRegionInfo &range_info) override {
    constexpr addr_t heap_start = 0x1000;
    constexpr addr_t stack_start = 0x10000000;
    if (load_addr < heap_start)
      return Status::FromErrorString("unmapped");
    if (load_addr < stack_start) {
      range_info.GetRange().SetRangeBase(heap_start);
      range_info.GetRange().SetRangeEnd(stack_start);
      range_info.SetIsStackMemory(eLazyBoolNo);
    } else {
      range_info.GetRange().SetRangeBase(stack_start);
      range_info.GetRange().SetRangeEnd(LLDB_INVALID_ADDRESS);
      range_info.SetIsStackMemory(eLazyBoolYes);
    }
    return {};
  }
};

class StackIDTest : public ::testing::Test {
  SubsystemRAII<FileSystem, HostInfo, PlatformMacOSX> subsystems;

public:
  void SetUp() override {
    ArchSpec arch("arm64-apple-macosx-");
    Platform::SetHostPlatform(
        PlatformRemoteMacOSX::CreateInstance(true, &arch));
    m_debugger_sp = Debugger::CreateInstance();
    ASSERT_TRUE(m_debugger_sp);
    PlatformSP platform_sp;
    ASSERT_TRUE(m_debugger_sp->GetTargetList()
                    .CreateTarget(*m_debugger_sp, "", arch, eLoadDependentsNo,
                                  platform_sp, m_target_sp)
                    .Success());
    ASSERT_TRUE(m_target_sp);
    m_process_sp = std::make_shared<MemoryRegionProcess>(
        m_target_sp, Listener::MakeListener("dummy"));
  }

protected:
  DebuggerSP m_debugger_sp;
  TargetSP m_target_sp;
  ProcessSP m_process_sp;
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

TEST_F(StackIDTest, IsCFAOnStack) {
  Process &process = *m_process_sp;
  StackID on_stack(0x1000, 0x7fff0000, nullptr, nullptr);
  StackID on_heap(0x1000, 0x2000000, nullptr, nullptr);
  StackID unmapped(0x1000, 0x100, nullptr, nullptr);

  EXPECT_TRUE(on_stack.IsCFAOnStack(process));
  EXPECT_FALSE(on_heap.IsCFAOnStack(process));
  // A call frame address the process can't classify counts as a stack address.
  EXPECT_TRUE(unmapped.IsCFAOnStack(process));
  EXPECT_TRUE(StackID().IsCFAOnStack(process));

  // Clearing a StackID forgets the cached answer.
  on_heap.Clear();
  EXPECT_TRUE(on_heap.IsCFAOnStack(process));
}
