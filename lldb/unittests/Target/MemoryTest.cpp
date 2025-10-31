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
#include <cstdint>

using namespace lldb_private;
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

/// A process class that, when asked to read memory from some address X, returns
/// the least significant byte of X.
class DummyReaderProcess : public Process {
public:
  // If true, `DoReadMemory` will not return all requested bytes.
  // It's not possible to control exactly how many bytes will be read, because
  // Process::ReadMemoryFromInferior tries to fulfill the entire request by
  // reading smaller chunks until it gets nothing back.
  bool read_less_than_requested = false;
  bool read_more_than_requested = false;

  size_t DoReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
                      Status &error) override {
    if (read_less_than_requested && size > 0)
      size--;
    if (read_more_than_requested)
      size *= 2;
    uint8_t *buffer = static_cast<uint8_t *>(buf);
    for (lldb::addr_t addr = vm_addr; addr < vm_addr + size; addr++)
      buffer[addr - vm_addr] = static_cast<uint8_t>(addr); // LSB of addr.
    return size;
  }
  // Boilerplate, nothing interesting below.
  DummyReaderProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}
  bool CanDebug(lldb::TargetSP, bool) override { return true; }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  bool DoUpdateThreadList(ThreadList &, ThreadList &) override { return false; }
  llvm::StringRef GetPluginName() override { return "Dummy"; }
};

TEST_F(MemoryTest, TestReadMemoryRanges) {
  ArchSpec arch("x86_64-apple-macosx-");

  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process_sp =
      std::make_shared<DummyReaderProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);

  {
    llvm::SmallVector<uint8_t, 0> buffer(1024, 0);
    // Read 8 ranges of 128 bytes with arbitrary base addresses.
    llvm::SmallVector<Range<addr_t, size_t>> ranges = {
        {0x12345, 128},      {0x11112222, 128}, {0x77777777, 128},
        {0xffaabbccdd, 128}, {0x0, 128},        {0x4242424242, 128},
        {0x17171717, 128},   {0x99999, 128}};

    llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> read_results =
        process_sp->ReadMemoryRanges(ranges, buffer);

    for (auto [range, memory] : llvm::zip(ranges, read_results)) {
      ASSERT_EQ(memory.size(), 128u);
      addr_t range_base = range.GetRangeBase();
      for (auto [idx, byte] : llvm::enumerate(memory))
        ASSERT_EQ(byte, static_cast<uint8_t>(range_base + idx));
    }
  }

  auto &dummy_process = static_cast<DummyReaderProcess &>(*process_sp);
  dummy_process.read_less_than_requested = true;
  {
    llvm::SmallVector<uint8_t, 0> buffer(1024, 0);
    llvm::SmallVector<Range<addr_t, size_t>> ranges = {
        {0x12345, 128}, {0x11112222, 128}, {0x77777777, 128}};
    llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> read_results =
        dummy_process.ReadMemoryRanges(ranges, buffer);
    for (auto [range, memory] : llvm::zip(ranges, read_results)) {
      ASSERT_LT(memory.size(), 128u);
      addr_t range_base = range.GetRangeBase();
      for (auto [idx, byte] : llvm::enumerate(memory))
        ASSERT_EQ(byte, static_cast<uint8_t>(range_base + idx));
    }
  }
}

using MemoryDeathTest = MemoryTest;

TEST_F(MemoryDeathTest, TestReadMemoryRangesReturnsTooMuch) {
  ArchSpec arch("x86_64-apple-macosx-");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);
  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process_sp =
      std::make_shared<DummyReaderProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);

  auto &dummy_process = static_cast<DummyReaderProcess &>(*process_sp);
  dummy_process.read_more_than_requested = true;
  llvm::SmallVector<uint8_t, 0> buffer(1024, 0);
  llvm::SmallVector<Range<addr_t, size_t>> ranges = {{0x12345, 128}};

  llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> read_results;
  ASSERT_DEBUG_DEATH(
      { read_results = process_sp->ReadMemoryRanges(ranges, buffer); },
      "read more than requested bytes");
#ifdef NDEBUG
  // With asserts off, the read should return empty ranges.
  ASSERT_EQ(read_results.size(), 1u);
  ASSERT_TRUE(read_results[0].empty());
#endif
}

TEST_F(MemoryDeathTest, TestReadMemoryRangesWithShortBuffer) {
  ArchSpec arch("x86_64-apple-macosx-");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);
  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process_sp =
      std::make_shared<DummyReaderProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);

  llvm::SmallVector<uint8_t, 0> short_buffer(10, 0);
  llvm::SmallVector<Range<addr_t, size_t>> ranges = {{0x12345, 128},
                                                     {0x11, 128}};
  llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> read_results;
  ASSERT_DEBUG_DEATH(
      { read_results = process_sp->ReadMemoryRanges(ranges, short_buffer); },
      "provided buffer is too short");
#ifdef NDEBUG
  // With asserts off, the read should return empty ranges.
  ASSERT_EQ(read_results.size(), ranges.size());
  for (llvm::MutableArrayRef<uint8_t> result : read_results)
    ASSERT_TRUE(result.empty());
#endif
}
