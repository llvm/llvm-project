//===-- MemoryTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Memory.h"
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <cstdint>

using namespace lldb_private;
using namespace lldb;

namespace {
class MockABI : public ABI {
public:
  // The only relevant method of this ABI:
  lldb::addr_t FixAnyAddress(lldb::addr_t pc) override {
    return pc & 0xf0ffffffffffffffULL;
  }

  explicit MockABI(ProcessSP process_sp)
      : ABI(std::move(process_sp), std::make_unique<llvm::MCRegisterInfo>()) {}
  static ABISP CreateInstance(ProcessSP process_sp, const ArchSpec &) {
    return std::make_shared<MockABI>(std::move(process_sp));
  }
  llvm::StringRef GetPluginName() override { return "mock"; }
  size_t GetRedZoneSize() const override { return 0; }
  bool PrepareTrivialCall(Thread &, addr_t, addr_t, addr_t,
                          llvm::ArrayRef<addr_t>) const override {
    return false;
  }
  bool GetArgumentValues(Thread &, ValueList &) const override { return false; }
  Status SetReturnValueObject(StackFrameSP &, ValueObjectSP &) override {
    return {};
  }
  UnwindPlanSP CreateFunctionEntryUnwindPlan() override { return nullptr; }
  UnwindPlanSP CreateDefaultUnwindPlan() override { return nullptr; }
  bool RegisterIsVolatile(const RegisterInfo *) override { return false; }
  bool CallFrameAddressIsValid(addr_t) override { return false; }
  bool CodeAddressIsValid(addr_t) override { return false; }
  void
  AugmentRegisterInfo(std::vector<DynamicRegisterInfo::Register> &) override {}

protected:
  ValueObjectSP GetReturnValueObjectImpl(Thread &,
                                         CompilerType &) const override {
    return nullptr;
  }
};

class MemoryTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    PlatformMacOSX::Initialize();
    PluginManager::RegisterPlugin("mock", "mock ABI", MockABI::CreateInstance);
  }
  void TearDown() override {
    PlatformMacOSX::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
    PluginManager::UnregisterPlugin(MockABI::CreateInstance);
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
  // Required by Target::ReadMemory() to call Process::ReadMemory()
  bool IsAlive() override { return true; }
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

    memset(buf, m_filler, num_bytes_to_write);
    return num_bytes_to_write;
  }
  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override {
    return false;
  }
  llvm::StringRef GetPluginName() override { return "Dummy"; }

  // Test-specific additions
  size_t m_bytes_left;
  int m_filler = 'B';
  MemoryCache &GetMemoryCache() { return m_memory_cache; }
  void SetMaxReadSize(size_t size) { m_bytes_left = size; }
  void SetFiller(int filler) { m_filler = filler; }
};

// A MemoryCache subclass that exposes the otherwise-protected L1 cache so a
// test can assert on the exact set of chunks it holds.
class TestMemoryCache : public MemoryCache {
public:
  using MemoryCache::MemoryCache;

  const BlockMap &GetL1Cache() const { return m_L1_cache; }
};
} // namespace

TargetSP CreateTarget(DebuggerSP &debugger_sp, ArchSpec &arch) {
  PlatformSP platform_sp;
  TargetSP target_sp;
  debugger_sp->GetTargetList().CreateTarget(
      *debugger_sp, "", arch, eLoadDependentsNo, platform_sp, target_sp);
  return target_sp;
}

static ProcessSP CreateProcess(lldb::TargetSP target_sp) {
  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process_sp = std::make_shared<DummyProcess>(target_sp, listener_sp);

  struct TargetHack : public Target {
    void SetProcess(ProcessSP process) { m_process_sp = process; }
  };
  static_cast<TargetHack *>(target_sp.get())->SetProcess(process_sp);

  return process_sp;
}

TEST_F(MemoryTest, TesetMemoryCacheRead) {
  ArchSpec arch("x86_64-apple-macosx-");

  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ProcessSP process_sp = CreateProcess(target_sp);
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

TEST_F(MemoryTest, TestL1Cache) {
  ArchSpec arch("arm64-apple-macosx");

  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ProcessSP process_sp = CreateProcess(target_sp);
  ASSERT_TRUE(process_sp);

  DummyProcess *process = static_cast<DummyProcess *>(process_sp.get());
  TestMemoryCache mem_cache(*process);

  auto add = [&](lldb::addr_t addr, size_t size, uint8_t fill) {
    mem_cache.AddL1CacheData(addr,
                             std::make_shared<DataBufferHeap>(size, fill));
  };

  // Asserts the L1 cache holds exactly `expected` chunks, matched by start
  // address, byte size, and a single repeated fill byte, in address order.
  struct Chunk {
    lldb::addr_t addr;
    size_t size;
    uint8_t fill;
  };
  auto expect_l1 = [&](std::vector<Chunk> expected) {
    const auto &l1 = mem_cache.GetL1Cache();
    ASSERT_EQ(l1.size(), expected.size());
    size_t i = 0;
    for (const auto &[addr, data_sp] : l1) {
      const Chunk &c = expected[i++];
      EXPECT_EQ(addr, c.addr);
      ASSERT_EQ(data_sp->GetByteSize(), c.size);
      const uint8_t *bytes = data_sp->GetBytes();
      for (size_t j = 0; j < c.size; ++j)
        EXPECT_EQ(bytes[j], c.fill)
            << "chunk 0x" << std::hex << addr << " byte " << std::dec << j;
    }
  };

  // Partial overlap: the new chunk overhangs the existing one on the right.
  mem_cache.Clear();
  add(0x1000, 0x100, 0xAA);
  add(0x1080, 0x100, 0xBB);
  expect_l1({{0x1000, 0x100, 0xAA}, {0x1080, 0x100, 0xBB}});

  // Partial overlap: the new chunk overhangs the existing one on the left.
  mem_cache.Clear();
  add(0x2080, 0x100, 0xAA);
  add(0x2000, 0x100, 0xBB);
  expect_l1({{0x2000, 0x100, 0xBB}, {0x2080, 0x100, 0xAA}});

  // New chunk fully contains an existing one: both are kept.
  mem_cache.Clear();
  add(0x3040, 0x40, 0xAA);
  add(0x3000, 0x100, 0xBB);
  expect_l1({{0x3000, 0x100, 0xBB}, {0x3040, 0x40, 0xAA}});

  // New chunk is fully contained by an existing one: both are kept.
  mem_cache.Clear();
  add(0x4000, 0x200, 0xAA);
  add(0x4080, 0x80, 0xBB);
  expect_l1({{0x4000, 0x200, 0xAA}, {0x4080, 0x80, 0xBB}});

  // New chunk partially overlaps two existing chunks; all three are kept.
  mem_cache.Clear();
  add(0x5000, 0x80, 0xAA);
  add(0x5100, 0x80, 0xCC);
  add(0x5040, 0x100, 0xBB);
  expect_l1(
      {{0x5000, 0x80, 0xAA}, {0x5040, 0x100, 0xBB}, {0x5100, 0x80, 0xCC}});

  // Disjoint chunks stay separate.
  mem_cache.Clear();
  add(0x6000, 0x80, 0xAA);
  add(0x6100, 0x80, 0xBB);
  expect_l1({{0x6000, 0x80, 0xAA}, {0x6100, 0x80, 0xBB}});

  // Adjacent (touching but not overlapping) chunks stay separate.
  mem_cache.Clear();
  add(0x7000, 0x80, 0xAA);
  add(0x7080, 0x80, 0xBB);
  expect_l1({{0x7000, 0x80, 0xAA}, {0x7080, 0x80, 0xBB}});

  // Flush must erase every chunk intersecting the flush range, including a
  // chunk that starts below the flushed address. Here 0x8140 lies only in the
  // lower-starting, longer chunk; it must be dropped while the chunk that does
  // not intersect survives untouched.
  mem_cache.Clear();
  add(0x8000, 0x180, 0xAA);
  add(0x8080, 0x40, 0xBB);
  mem_cache.Flush(0x8140, 0x4);
  expect_l1({{0x8080, 0x40, 0xBB}});

  // A flush intersecting several partially overlapping chunks drops all of
  // them, while a chunk it does not intersect is left in place.
  mem_cache.Clear();
  add(0x9000, 0x80, 0xAA);
  add(0x9040, 0x100, 0xBB);
  add(0x9100, 0x80, 0xCC);
  mem_cache.Flush(0x9060, 0x1);
  expect_l1({{0x9100, 0x80, 0xCC}});
}

TEST_F(MemoryTest, TestReadInteger) {
  ArchSpec arch("x86_64-apple-macosx-");

  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ProcessSP process_sp = CreateProcess(target_sp);
  ASSERT_TRUE(process_sp);

  DummyProcess *process = static_cast<DummyProcess *>(process_sp.get());
  Status error;

  process->SetFiller(0xff);
  process->SetMaxReadSize(256);
  // The ReadSignedIntegerFromMemory() methods return int64_t. Check that they
  // extend the sign correctly when reading 32-bit values.
  EXPECT_EQ(-1,
            target_sp->ReadSignedIntegerFromMemory(Address(0), 4, 0, error));
  EXPECT_EQ(-1, process->ReadSignedIntegerFromMemory(0, 4, 0, error));
  // Check reading 64-bit values as well.
  EXPECT_EQ(-1,
            target_sp->ReadSignedIntegerFromMemory(Address(0), 8, 0, error));
  EXPECT_EQ(-1, process->ReadSignedIntegerFromMemory(0, 8, 0, error));

  // ReadUnsignedIntegerFromMemory() should not extend the sign.
  EXPECT_EQ(0xffffffffULL,
            target_sp->ReadUnsignedIntegerFromMemory(Address(0), 4, 0, error));
  EXPECT_EQ(0xffffffffULL,
            process->ReadUnsignedIntegerFromMemory(0, 4, 0, error));
  EXPECT_EQ(0xffffffffffffffffULL,
            target_sp->ReadUnsignedIntegerFromMemory(Address(0), 8, 0, error));
  EXPECT_EQ(0xffffffffffffffffULL,
            process->ReadUnsignedIntegerFromMemory(0, 8, 0, error));

  // Check reading positive values.
  process->GetMemoryCache().Clear();
  process->SetFiller(0x7f);
  process->SetMaxReadSize(256);
  EXPECT_EQ(0x7f7f7f7fLL,
            target_sp->ReadSignedIntegerFromMemory(Address(0), 4, 0, error));
  EXPECT_EQ(0x7f7f7f7fLL, process->ReadSignedIntegerFromMemory(0, 4, 0, error));
  EXPECT_EQ(0x7f7f7f7f7f7f7f7fLL,
            target_sp->ReadSignedIntegerFromMemory(Address(0), 8, 0, error));
  EXPECT_EQ(0x7f7f7f7f7f7f7f7fLL,
            process->ReadSignedIntegerFromMemory(0, 8, 0, error));
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
        {0x6789, 128}, {0x333344444, 128}, {0x99999999, 128}};
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
  // gtest death-tests execute in a sub-process (fork), which invalidates
  // any signpost handles and would cause spurious crashes if used. Use the
  // "threadsafe" style of death-test to work around this.
  // FIXME: we should set this only if signposts are enabled, and do so
  // for the entire test-suite.
  GTEST_FLAG_SET(death_test_style, "threadsafe");

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
  // gtest death-tests execute in a sub-process (fork), which invalidates
  // any signpost handles and would cause spurious crashes if used. Use the
  // "threadsafe" style of death-test to work around this.
  // FIXME: we should set this only if signposts are enabled, and do so
  // for the entire test-suite.
  GTEST_FLAG_SET(death_test_style, "threadsafe");

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

/// A process class whose memory contains the following map of addresses to
/// strings:
///   100 -> "hello\0"
///   200 -> "\0"
///   201 -> "goodbye"
///   300 -> a string composed of 500 'c' characters, followed by '\0'.
///   addresses >= 1024 -> error
class StringReaderProcess : public Process {
public:
  char memory[1024];
  void initialize_memory() {
    // Use some easily identifiable character for the areas of memory we're not
    // intending to read.
    memset(memory, '?', 1024);
    strcpy(&memory[100], "hello");
    strcpy(&memory[200], "");
    strcpy(&memory[201], "goodbye");
    std::vector<char> long_str(500, 'c');
    long_str.push_back('\0');
    strcpy(&memory[300], long_str.data());
  }

  size_t DoReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
                      Status &error) override {
    if (vm_addr >= 1024) {
      error = Status::FromErrorString("out of bounds!");
      return 0;
    }
    memcpy(buf, memory + vm_addr, size);
    return size;
  }
  StringReaderProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {
    initialize_memory();
  }
  // Boilerplate, nothing interesting below.
  bool CanDebug(lldb::TargetSP, bool) override { return true; }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  bool DoUpdateThreadList(ThreadList &, ThreadList &) override { return false; }
  llvm::StringRef GetPluginName() override { return "Dummy"; }
};

TEST_F(MemoryTest, TestReadCStringsFromMemory) {
  ArchSpec arch("x86_64-apple-macosx-");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);
  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process_sp =
      std::make_shared<StringReaderProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process_sp);

  // See the docs for StringReaderProcess above for an explanation of these
  // addresses.
  llvm::SmallVector<std::optional<std::string>> maybe_strings =
      process_sp->ReadCStringsFromMemory({100, 200, 201, 300, 0xffffff});
  ASSERT_EQ(maybe_strings.size(), 5ull);
  auto expected_valid_strings = llvm::ArrayRef(maybe_strings).take_front(4);

  std::vector<char> long_str(500, 'c');
  long_str.push_back('\0');
  std::string big_str(long_str.data());

  const std::vector<std::optional<std::string>> expected_answers = {
      "hello", "", "goodbye", big_str, std::nullopt};
  for (auto [maybe_str, expected_answer] :
       llvm::zip(expected_valid_strings, expected_answers))
    EXPECT_EQ(maybe_str, expected_answer);
}

TEST_F(MemoryTest, TestReadPointersFromMemory) {
  ArchSpec arch("x86_64-apple-macosx-");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);
  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process =
      std::make_shared<DummyReaderProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process);

  // Read pointers at arbitrary addresses.
  llvm::SmallVector<addr_t> ptr_locs = {0x0, 0x100, 0x2000, 0x123400};
  // Because of how DummyReaderProcess works, each byte of a memory read result
  // is its address modulo 256:
  constexpr addr_t expected_result = 0x0706050403020100;

  llvm::SmallVector<std::optional<addr_t>> read_results =
      process->ReadPointersFromMemory(ptr_locs);

  for (std::optional<addr_t> maybe_ptr : read_results) {
    ASSERT_TRUE(maybe_ptr.has_value());
    EXPECT_EQ(*maybe_ptr, expected_result);
  }
}

TEST_F(MemoryTest, TestReadUnsignedIntegersFromMemory) {
  ArchSpec arch("x86_64-apple-macosx-");

  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);
  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process =
      std::make_shared<DummyReaderProcess>(target_sp, listener_sp);
  ASSERT_TRUE(process);

  { // Test reads of size 1
    llvm::SmallVector<addr_t> locs = {0x0, 0x101, 0x2002, 0x123403};
    llvm::SmallVector<std::optional<addr_t>> read_results =
        process->ReadUnsignedIntegersFromMemory(locs, /*byte_size=*/1);

    for (auto [maybe_int, loc] : llvm::zip(read_results, locs)) {
      ASSERT_TRUE(maybe_int.has_value());
      EXPECT_EQ(*maybe_int, static_cast<uint8_t>(loc));
    }
  }

  { // Test reads of size 2
    llvm::SmallVector<addr_t> locs = {0x0, 0x101, 0x2002, 0x123403};
    llvm::SmallVector<std::optional<addr_t>> read_results =
        process->ReadUnsignedIntegersFromMemory(locs, /*byte_size=*/2);

    for (auto [maybe_int, loc] : llvm::zip(read_results, locs)) {
      ASSERT_TRUE(maybe_int.has_value());
      uint64_t lsb = static_cast<uint8_t>(loc);
      uint64_t expected_result = ((lsb + 1) << 8) | lsb;
      EXPECT_EQ(*maybe_int, expected_result);
    }
  }

  { // Test reads of size 4
    llvm::SmallVector<addr_t> locs = {0x0, 0x101, 0x2002, 0x123403};
    llvm::SmallVector<std::optional<addr_t>> read_results =
        process->ReadUnsignedIntegersFromMemory(locs, /*byte_size=*/4);

    for (auto [maybe_int, loc] : llvm::zip(read_results, locs)) {
      ASSERT_TRUE(maybe_int.has_value());
      uint64_t lsb = static_cast<uint8_t>(loc);
      uint64_t expected_result =
          ((lsb + 3) << 24) | ((lsb + 2) << 16) | ((lsb + 1) << 8) | lsb;
      EXPECT_EQ(*maybe_int, expected_result);
    }
  }

  { // Test reads of size 8
    llvm::SmallVector<addr_t> locs = {0x0, 0x101, 0x2002, 0x123403};
    llvm::SmallVector<std::optional<addr_t>> read_results =
        process->ReadUnsignedIntegersFromMemory(locs, /*byte_size=*/8);

    for (auto [maybe_int, loc] : llvm::zip(read_results, locs)) {
      ASSERT_TRUE(maybe_int.has_value());
      uint64_t lsb = static_cast<uint8_t>(loc);
      uint64_t expected_result = ((lsb + 7) << 56) | ((lsb + 6) << 48) |
                                 ((lsb + 5) << 40) | ((lsb + 4) << 32) |
                                 ((lsb + 3) << 24) | ((lsb + 2) << 16) |
                                 ((lsb + 1) << 8) | lsb;
      EXPECT_EQ(*maybe_int, expected_result);
    }
  }
}

// A process that, when asked to read memory from address X, returns the top
// byte of X.
class DummyMSBReaderProcess : public Process {
public:
  // Only call this method with exactly one range.
  llvm::SmallVector<llvm::MutableArrayRef<uint8_t>>
  DoReadMemoryRanges(llvm::ArrayRef<Range<addr_t, size_t>> ranges,
                     llvm::MutableArrayRef<uint8_t> buffer) override {
    buffer[0] = static_cast<uint8_t>(ranges[0].GetRangeBase() >> 56);
    return {{buffer.take_front(1)}};
  }
  // Boilerplate, nothing interesting below.
  DummyMSBReaderProcess(TargetSP target_sp, ListenerSP listener_sp)
      : Process(target_sp, listener_sp) {}
  bool CanDebug(TargetSP, bool) override { return true; }
  Status DoDestroy() override { return {}; }
  void RefreshStateAfterStop() override {}
  bool DoUpdateThreadList(ThreadList &, ThreadList &) override { return false; }
  llvm::StringRef GetPluginName() override { return "Dummy"; }
  size_t DoReadMemory(addr_t, void *, size_t, Status &) override {
    llvm_unreachable("don't call this");
  }
};

TEST_F(MemoryTest, TestReadMemoryRangesClearMetadata) {
  ArchSpec arch("x86_64-apple-macosx-");

  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);
  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  ProcessSP process_sp =
      std::make_shared<DummyMSBReaderProcess>(target_sp, listener_sp);

  llvm::SmallVector<uint8_t, 0> buffer(1024, 0);
  llvm::SmallVector<Range<addr_t, size_t>> ranges = {{0xff0123456789abcd, 1}};
  llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> read_results =
      process_sp->ReadMemoryRanges(ranges, buffer);
  ASSERT_EQ(read_results.size(), 1ull);
  ASSERT_EQ(read_results[0].size(), 1ull);
  ASSERT_EQ(read_results[0][0], 0xf0); // The ABI masks with 0xf0.
}

// The live process read fails outright, so Target::ReadMemory must fall all the
// way through to the file-cache fallback at the end of the function, which
// serves the bytes out of the module's (__DATA,__data) section. A full read
// there must report success, and a short read must not.
TEST_F(MemoryTest, TestReadMemoryClearsStaleError) {
  SubsystemRAII<ObjectFileMachO> subsystems;

  ArchSpec arch("x86_64-apple-macosx-");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ProcessSP process_sp = CreateProcess(target_sp);
  ASSERT_TRUE(process_sp);

  // The process can't produce a single byte, so the read must fail.
  static_cast<DummyProcess *>(process_sp.get())->SetMaxReadSize(0);

  auto expected_file = TestFile::fromYaml(R"(
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x1000007
  cpusubtype:      0x3
  filetype:        0x2
  ncmds:           1
  sizeofcmds:      152
  flags:           0x200085
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         152
    segname:         __DATA
    vmaddr:          0x100001000
    vmsize:          0xC
    fileoff:         0x1000
    filesize:        0xC
    maxprot:         3
    initprot:        3
    nsects:          1
    flags:           0
    Sections:
      - sectname:        __data
        segname:         __DATA
        addr:            0x100001000
        size:            12
        offset:          0x1000
        align:           0
        reloff:          0x0
        nreloc:          0
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         68656C6C6F20776F726C6400
...
)");
  // "expected_file" owns the buffer the Module reads through, so it has to
  // outlive every ReadMemory() call below.
  ASSERT_THAT_EXPECTED(expected_file, llvm::Succeeded());

  ModuleSP module_sp = std::make_shared<Module>(expected_file->moduleSpec());
  target_sp->GetImages().Append(module_sp, /*notify=*/false);

  SectionList *sections = module_sp->GetSectionList();
  ASSERT_TRUE(sections);
  SectionSP section_sp = sections->FindSectionByName(ConstString("__data"));
  ASSERT_TRUE(section_sp);
  target_sp->SetSectionLoadAddress(section_sp, section_sp->GetFileAddress());

  // force_live_memory = true skips the read-only file-cache fast path near the
  // top of ReadMemory, and the section is writable so that path would reject
  // it anyway. The fallback at the end is the only thing that can serve this.
  Address addr;
  ASSERT_TRUE(
      target_sp->ResolveLoadAddress(section_sp->GetFileAddress(), addr));
  char buf[5] = {};
  Status error;
  size_t bytes_read = target_sp->ReadMemory(addr, buf, sizeof(buf), error,
                                            /*force_live_memory=*/true);
  ASSERT_EQ(bytes_read, sizeof(buf));
  EXPECT_TRUE(error.Success()) << error.AsCString();
  EXPECT_EQ(llvm::StringRef(buf, sizeof(buf)), "hello");

  // A short read must be reported as an error.
  char big[20] = {};
  Status short_error;
  EXPECT_EQ(target_sp->ReadMemory(addr, big, sizeof(big), short_error,
                                  /*force_live_memory=*/true),
            12u);
  EXPECT_TRUE(short_error.Fail());
}
