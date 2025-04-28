//===-- MemoryTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Memory.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb_private::repro;
using namespace lldb;

namespace {
class MemoryTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    PlatformMacOSX::Initialize();
  }
  void TearDown() override {
    PlatformMacOSX::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};

class DummyProcess : public Process {
public:
  DummyProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp)
      : Process(target_sp, listener_sp), m_bytes_left(0) {}

  // Required overrides
  bool CanDebug(lldb::TargetSP target, bool plugin_specified_by_name) override {
    return true;
  }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  size_t DoReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
                      Status &error) override {
    if (m_bytes_left == 0)
      return 0;

    size_t num_bytes_to_write = size;
    if (m_bytes_left < size) {
      num_bytes_to_write = m_bytes_left;
      m_bytes_left = 0;
    } else {
      m_bytes_left -= size;
    }

    memset(buf, 'B', num_bytes_to_write);
    return num_bytes_to_write;
  }
  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override {
    return false;
  }
  llvm::StringRef GetPluginName() override { return "Dummy"; }

  // Test-specific additions
  size_t m_bytes_left;
  MemoryCache &GetMemoryCache() { return m_memory_cache; }
  void SetMaxReadSize(size_t size) { m_bytes_left = size; }
};
} // namespace

TargetSP CreateTarget(DebuggerSP &debugger_sp, ArchSpec &arch) {
  PlatformSP platform_sp;
  TargetSP target_sp;
  debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);
  return target_sp;
}

TEST_F(MemoryTest, TesetMemoryCacheRead) {
  ArchSpec arch("x86_64-apple-macosx-");

  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process_sp = std::make_shared<DummyProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);

  DummyProcess *process = static_cast<DummyProcess *>(process_sp.get());
  MemoryCache &mem_cache = process->GetMemoryCache();
  const uint64_t l2_cache_size = process->GetMemoryCacheLineSize();
  Status error;
  auto data_sp = std::make_shared<DataBufferHeap>(l2_cache_size * 2, '\0');
  size_t bytes_read = 0;

  // Cache empty, memory read fails, size > l2 cache size
  process->SetMaxReadSize(0);
  bytes_read = mem_cache.Read(0x1000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == 0);

  // Cache empty, memory read fails, size <= l2 cache size
  data_sp->SetByteSize(l2_cache_size);
  bytes_read = mem_cache.Read(0x1000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == 0);

  // Cache empty, memory read succeeds, size > l2 cache size
  process->SetMaxReadSize(l2_cache_size * 4);
  data_sp->SetByteSize(l2_cache_size * 2);
  bytes_read = mem_cache.Read(0x1000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == data_sp->GetByteSize());
  ASSERT_TRUE(process->m_bytes_left == l2_cache_size * 2);

  // Reading data previously cached (not in L2 cache).
  data_sp->SetByteSize(l2_cache_size + 1);
  bytes_read = mem_cache.Read(0x1000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == data_sp->GetByteSize());
  ASSERT_TRUE(process->m_bytes_left == l2_cache_size * 2); // Verify we didn't
                                                           // read from the
                                                           // inferior.

  // Read from a different address, but make the size == l2 cache size.
  // This should fill in a the L2 cache.
  data_sp->SetByteSize(l2_cache_size);
  bytes_read = mem_cache.Read(0x2000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == data_sp->GetByteSize());
  ASSERT_TRUE(process->m_bytes_left == l2_cache_size);

  // Read from that L2 cache entry but read less than size of the cache line.
  // Additionally, read from an offset.
  data_sp->SetByteSize(l2_cache_size - 5);
  bytes_read = mem_cache.Read(0x2001, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == data_sp->GetByteSize());
  ASSERT_TRUE(process->m_bytes_left == l2_cache_size); // Verify we didn't read
                                                       // from the inferior.

  // What happens if we try to populate an L2 cache line but the read gives less
  // than the size of a cache line?
  process->SetMaxReadSize(l2_cache_size - 10);
  data_sp->SetByteSize(l2_cache_size - 5);
  bytes_read = mem_cache.Read(0x3000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == l2_cache_size - 10);
  ASSERT_TRUE(process->m_bytes_left == 0);

  // What happens if we have a partial L2 cache line filled in and we try to
  // read the part that isn't filled in?
  data_sp->SetByteSize(10);
  bytes_read = mem_cache.Read(0x3000 + l2_cache_size - 10, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == 0); // The last 10 bytes from this line are
                                // missing and we should be reading nothing
                                // here.

  // What happens when we try to straddle 2 cache lines?
  process->SetMaxReadSize(l2_cache_size * 2);
  data_sp->SetByteSize(l2_cache_size);
  bytes_read = mem_cache.Read(0x4001, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == l2_cache_size);
  ASSERT_TRUE(process->m_bytes_left == 0);

  // What happens when we try to straddle 2 cache lines where the first one is
  // only partially filled?
  process->SetMaxReadSize(l2_cache_size - 1);
  data_sp->SetByteSize(l2_cache_size);
  bytes_read = mem_cache.Read(0x5005, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == l2_cache_size - 6); // Ignoring the first 5 bytes,
                                                // missing the last byte
  ASSERT_TRUE(process->m_bytes_left == 0);

  // What happens if we add an invalid range and try to do a read larger than
  // a cache line?
  mem_cache.AddInvalidRange(0x6000, l2_cache_size * 2);
  process->SetMaxReadSize(l2_cache_size * 2);
  data_sp->SetByteSize(l2_cache_size * 2);
  bytes_read = mem_cache.Read(0x6000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == 0);
  ASSERT_TRUE(process->m_bytes_left == l2_cache_size * 2);

  // What happens if we add an invalid range and try to do a read lt/eq a
  // cache line?
  mem_cache.AddInvalidRange(0x7000, l2_cache_size);
  process->SetMaxReadSize(l2_cache_size);
  data_sp->SetByteSize(l2_cache_size);
  bytes_read = mem_cache.Read(0x7000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == 0);
  ASSERT_TRUE(process->m_bytes_left == l2_cache_size);

  // What happens if we remove the invalid range and read again?
  mem_cache.RemoveInvalidRange(0x7000, l2_cache_size);
  bytes_read = mem_cache.Read(0x7000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == l2_cache_size);
  ASSERT_TRUE(process->m_bytes_left == 0);

  // What happens if we flush and read again?
  process->SetMaxReadSize(l2_cache_size * 2);
  mem_cache.Flush(0x7000, l2_cache_size);
  bytes_read = mem_cache.Read(0x7000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == l2_cache_size);
  ASSERT_TRUE(process->m_bytes_left == l2_cache_size); // Verify that we re-read
                                                       // instead of using an
                                                       // old cache
}
