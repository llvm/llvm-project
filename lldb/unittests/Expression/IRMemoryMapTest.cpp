//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/IRMemoryMap.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/Process.h"
#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

namespace {

/// A process that reports CanJIT() = false and IsAlive() = true, with a
/// GetMemoryRegionInfo that succeeds for the first region but fails beyond
/// it (mimicking targets like WebAssembly).
class NoJITProcess : public Process {
public:
  NoJITProcess(TargetSP target_sp, ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {
    SetCanJIT(false);
  }

  bool CanDebug(TargetSP target, bool plugin_specified_by_name) override {
    return true;
  }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  size_t DoReadMemory(addr_t vm_addr, void *buf, size_t size,
                      Status &error) override {
    return 0;
  }
  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override {
    return false;
  }
  llvm::StringRef GetPluginName() override { return "no-jit-process"; }

  bool IsAlive() override { return true; }

  Status DoGetMemoryRegionInfo(addr_t load_addr,
                               MemoryRegionInfo &range_info) override {
    // Report the first region, but fail for anything beyond it. This
    // simulates a target whose address space is not fully queryable.
    if (load_addr < 0x10000) {
      range_info.GetRange().SetRangeBase(0);
      range_info.GetRange().SetByteSize(0x10000);
      range_info.SetReadable(eLazyBoolYes);
      range_info.SetWritable(eLazyBoolYes);
      range_info.SetExecutable(eLazyBoolNo);
      return Status();
    }
    return Status::FromErrorString(
        "memory region info unavailable past linear memory");
  }
};

/// Expose the protected GetProcessWP so we can inject a mock process.
class TestIRMemoryMap : public IRMemoryMap {
public:
  using IRMemoryMap::IRMemoryMap;
  void SetProcess(ProcessSP process_sp) { GetProcessWP() = process_sp; }
};

class IRMemoryMapTest : public ::testing::Test {
public:
  SubsystemRAII<FileSystem, HostInfo, platform_linux::PlatformLinux> subsystem;
};

} // namespace

// Verify that host-only allocations succeed when the process is alive but
// can't JIT. Before the fix, FindSpace would probe GetMemoryRegionInfo,
// which could assert/crash for targets with partial address-space coverage.
TEST_F(IRMemoryMapTest, FindSpaceNoJIT) {
  ArchSpec arch("i386-pc-linux");
  Platform::SetHostPlatform(
      platform_linux::PlatformLinux::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp;
  PlatformSP platform_sp;
  Status error = debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);
  ASSERT_TRUE(target_sp);

  ListenerSP listener_sp(Listener::MakeListener("test"));
  auto process_sp = std::make_shared<NoJITProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);
  ASSERT_FALSE(process_sp->CanJIT());
  ASSERT_TRUE(process_sp->IsAlive());

  TestIRMemoryMap memory_map(target_sp);
  memory_map.SetProcess(process_sp);

  // This would previously crash in FindSpace via lldbassert when
  // GetMemoryRegionInfo succeeded then failed on a subsequent call.
  auto addr_or_err =
      memory_map.Malloc(1024, 8, ePermissionsReadable | ePermissionsWritable,
                        IRMemoryMap::eAllocationPolicyHostOnly, false);
  ASSERT_THAT_EXPECTED(addr_or_err, llvm::Succeeded());
  EXPECT_NE(*addr_or_err, LLDB_INVALID_ADDRESS);

  // A second allocation should also succeed and not overlap.
  auto addr2_or_err =
      memory_map.Malloc(2048, 8, ePermissionsReadable | ePermissionsWritable,
                        IRMemoryMap::eAllocationPolicyHostOnly, false);
  ASSERT_THAT_EXPECTED(addr2_or_err, llvm::Succeeded());
  EXPECT_NE(*addr2_or_err, LLDB_INVALID_ADDRESS);
  EXPECT_NE(*addr_or_err, *addr2_or_err);

  Debugger::Destroy(debugger_sp);
}
