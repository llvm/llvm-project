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
#include <utility>

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
  size_t DoReadMemory(const ProcessAddress &process_addr, void *buf,
                      size_t size, Status &error) override {
    m_reads.emplace_back(process_addr.GetValue(), size);
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
  // Every DoReadMemory request, as (address, size).
  llvm::SmallVector<std::pair<lldb::addr_t, size_t>, 4> m_reads;
  MemoryCache &GetMemoryCache() { return m_memory_cache; }
  void SetMaxReadSize(size_t size) { m_bytes_left = size; }
  void SetFiller(int filler) { m_filler = filler; }
};

// A MemoryCache subclass that exposes the otherwise-protected caches so a
// test can assert on the exact set of entries they hold.
class TestMemoryCache : public MemoryCache {
public:
  using MemoryCache::MemoryCache;

  const ChunkCache &GetL1Cache() const { return m_L1_cache; }
  const LineCache &GetL2Cache() const { return m_L2_cache; }
};

using CacheEntries =
    std::vector<std::pair<lldb::addr_t, llvm::ArrayRef<uint8_t>>>;

// The chunks of \a cache, which iterates in address order already.
CacheEntries Snapshot(const ChunkCache &cache) {
  CacheEntries entries;
  for (const auto &[addr, chunk] : cache)
    entries.emplace_back(addr, llvm::ArrayRef(chunk));
  return entries;
}

// The lines of \a cache in address order, which its iteration does not give.
CacheEntries Snapshot(const LineCache &cache) {
  const uint32_t line_size = cache.GetLineByteSize();
  CacheEntries entries;
  for (const auto &[line_idx, line] : cache)
    entries.emplace_back(line_idx * line_size,
                         llvm::ArrayRef(line.get(), line_size));
  llvm::sort(entries, llvm::less_first());
  return entries;
}
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

// Builds the debugger, target and process a cache test needs and keeps them
// alive.
namespace {
class CacheTestProcess {
public:
  explicit CacheTestProcess(llvm::StringRef triple = "arm64-apple-macosx")
      : m_arch(triple) {
    Platform::SetHostPlatform(
        PlatformRemoteMacOSX::CreateInstance(true, &m_arch));
    m_debugger_sp = Debugger::CreateInstance();
    if (!m_debugger_sp)
      return;
    m_target_sp = CreateTarget(m_debugger_sp, m_arch);
    if (!m_target_sp)
      return;
    m_process_sp = CreateProcess(m_target_sp);
    m_process = static_cast<DummyProcess *>(m_process_sp.get());
  }

  DummyProcess *GetProcess() const { return m_process; }
  uint64_t GetLineSize() const { return m_process->GetMemoryCacheLineSize(); }

private:
  ArchSpec m_arch;
  lldb::DebuggerSP m_debugger_sp;
  lldb::TargetSP m_target_sp;
  lldb::ProcessSP m_process_sp;
  DummyProcess *m_process = nullptr;
};

void AddCacheChunk(TestMemoryCache &cache, lldb::addr_t addr, size_t size,
                   uint8_t fill) {
  cache.AddCacheData(addr, std::make_shared<DataBufferHeap>(size, fill));
}

bool AllBytesAre(llvm::ArrayRef<uint8_t> bytes, uint8_t fill) {
  return llvm::all_of(bytes, [fill](uint8_t byte) { return byte == fill; });
}
} // namespace

TEST_F(MemoryTest, TesetMemoryCacheRead) {
  CacheTestProcess proc("x86_64-apple-macosx-");
  ASSERT_TRUE(proc.GetProcess());
  DummyProcess *process = proc.GetProcess();
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
  process->m_reads.clear();
  bytes_read = mem_cache.Read(0x1000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == data_sp->GetByteSize());
  // A read larger than a line goes to the inferior as asked, not rounded to a
  // line.
  ASSERT_EQ(process->m_reads.size(), 1u);
  EXPECT_EQ(process->m_reads[0].first, 0x1000u);
  EXPECT_EQ(process->m_reads[0].second, l2_cache_size * 2);

  // Reading data previously cached (not in L2 cache).
  data_sp->SetByteSize(l2_cache_size + 1);
  process->m_reads.clear();
  bytes_read = mem_cache.Read(0x1000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == data_sp->GetByteSize());
  EXPECT_TRUE(process->m_reads.empty());

  // Read from a different address, but make the size == l2 cache size.
  // This should fill in a the L2 cache.
  data_sp->SetByteSize(l2_cache_size);
  process->m_reads.clear();
  bytes_read = mem_cache.Read(0x2000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == data_sp->GetByteSize());
  ASSERT_EQ(process->m_reads.size(), 1u);
  EXPECT_EQ(process->m_reads[0].first, 0x2000u);
  EXPECT_EQ(process->m_reads[0].second, l2_cache_size);

  // Read from that L2 cache entry but read less than size of the cache line.
  // Additionally, read from an offset.
  data_sp->SetByteSize(l2_cache_size - 5);
  process->m_reads.clear();
  bytes_read = mem_cache.Read(0x2001, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == data_sp->GetByteSize());
  EXPECT_TRUE(process->m_reads.empty());

  // What happens if we try to populate an L2 cache line but the read gives less
  // than the size of a cache line?
  process->SetMaxReadSize(l2_cache_size - 10);
  data_sp->SetByteSize(l2_cache_size - 5);
  process->m_reads.clear();
  bytes_read = mem_cache.Read(0x3000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == l2_cache_size - 10);
  EXPECT_TRUE(error.Success());
  ASSERT_EQ(process->m_reads.size(), 2u);
  EXPECT_EQ(process->m_reads[0].first, 0x3000u);
  EXPECT_EQ(process->m_reads[0].second, l2_cache_size);
  EXPECT_EQ(process->m_reads[1].first, 0x3000u + l2_cache_size - 10);
  EXPECT_EQ(process->m_reads[1].second, 10u);

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
  process->m_reads.clear();
  bytes_read = mem_cache.Read(0x4001, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == l2_cache_size);
  // One request, both lines whole.
  ASSERT_EQ(process->m_reads.size(), 1u);
  EXPECT_EQ(process->m_reads[0].first, 0x4000u);
  EXPECT_EQ(process->m_reads[0].second, 2 * l2_cache_size);

  // What happens when we try to straddle 2 cache lines where the first one is
  // only partially filled?
  process->SetMaxReadSize(l2_cache_size - 1);
  data_sp->SetByteSize(l2_cache_size);
  process->m_reads.clear();
  bytes_read = mem_cache.Read(0x5005, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == l2_cache_size - 6); // Ignoring the first 5 bytes,
                                                // missing the last byte
  ASSERT_TRUE(error.Success());
  // The request is grown to both lines it touches and starts at the line base,
  // so it spends part of the mock's byte budget below the read.
  ASSERT_EQ(process->m_reads.size(), 2u);
  EXPECT_EQ(process->m_reads[0].first, 0x5000u);
  EXPECT_EQ(process->m_reads[0].second, 2 * l2_cache_size);
  EXPECT_EQ(process->m_reads[1].first, 0x5000u + l2_cache_size - 1);
  EXPECT_EQ(process->m_reads[1].second, l2_cache_size + 1);

  // What happens if we add an invalid range and try to do a read larger than
  // a cache line?
  mem_cache.AddInvalidRange(0x6000, l2_cache_size * 2);
  process->SetMaxReadSize(l2_cache_size * 2);
  data_sp->SetByteSize(l2_cache_size * 2);
  process->m_reads.clear();
  bytes_read = mem_cache.Read(0x6000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == 0);
  EXPECT_TRUE(process->m_reads.empty());

  // What happens if we add an invalid range and try to do a read lt/eq a
  // cache line?
  mem_cache.AddInvalidRange(0x7000, l2_cache_size);
  process->SetMaxReadSize(l2_cache_size);
  data_sp->SetByteSize(l2_cache_size);
  process->m_reads.clear();
  bytes_read = mem_cache.Read(0x7000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == 0);
  EXPECT_TRUE(process->m_reads.empty());

  // What happens if we remove the invalid range and read again?
  mem_cache.RemoveInvalidRange(0x7000, l2_cache_size);
  process->m_reads.clear();
  bytes_read = mem_cache.Read(0x7000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == l2_cache_size);
  ASSERT_EQ(process->m_reads.size(), 1u);
  EXPECT_EQ(process->m_reads[0].first, 0x7000u);
  EXPECT_EQ(process->m_reads[0].second, l2_cache_size);

  // What happens if we flush and read again?
  process->SetMaxReadSize(l2_cache_size * 2);
  mem_cache.Flush(0x7000, l2_cache_size);
  process->m_reads.clear();
  bytes_read = mem_cache.Read(0x7000, data_sp->GetBytes(),
                              data_sp->GetByteSize(), error);
  ASSERT_TRUE(bytes_read == l2_cache_size);
  // Verify that we re-read instead of using an old cache.
  ASSERT_EQ(process->m_reads.size(), 1u);
  EXPECT_EQ(process->m_reads[0].first, 0x7000u);
  EXPECT_EQ(process->m_reads[0].second, l2_cache_size);
}

TEST_F(MemoryTest, TestCachePartition) {
  CacheTestProcess proc;
  ASSERT_TRUE(proc.GetProcess());
  DummyProcess *process = proc.GetProcess();
  TestMemoryCache mem_cache(*process);
  const lldb::addr_t line = process->GetMemoryCacheLineSize();
  ASSERT_EQ(line, 512u);

  auto add = [&](lldb::addr_t addr, size_t size, uint8_t fill) {
    AddCacheChunk(mem_cache, addr, size, fill);
  };

  // Asserts a snapshot holds exactly `expected` entries, in address order.
  struct Chunk {
    lldb::addr_t addr;
    size_t size;
    uint8_t fill;
  };
  auto expect = [](const CacheEntries &entries, std::vector<Chunk> expected) {
    ASSERT_EQ(entries.size(), expected.size());
    size_t i = 0;
    for (const auto &[addr, bytes] : entries) {
      const Chunk &c = expected[i++];
      EXPECT_EQ(addr, c.addr);
      ASSERT_EQ(bytes.size(), c.size);
      for (size_t j = 0; j < c.size; ++j)
        EXPECT_EQ(bytes[j], c.fill)
            << "chunk 0x" << std::hex << addr << " byte " << std::dec << j;
    }
  };
  auto expect_l1 = [&](std::vector<Chunk> expected) {
    expect(Snapshot(mem_cache.GetL1Cache()), expected);
  };
  auto expect_l2 = [&](std::vector<Chunk> expected) {
    expect(Snapshot(mem_cache.GetL2Cache()), expected);
  };

  // Partial overlap: only the part not already held is added on the right.
  mem_cache.Clear();
  add(0x1000, 0x100, 0xAA);
  add(0x1080, 0x100, 0xBB);
  expect_l1({{0x1000, 0x100, 0xAA}, {0x1100, 0x80, 0xBB}});
  expect_l2({});

  // Partial overlap: the new chunk is added on the left.
  mem_cache.Clear();
  add(0x2080, 0x100, 0xAA);
  add(0x2000, 0x100, 0xBB);
  expect_l1({{0x2000, 0x80, 0xBB}, {0x2080, 0x100, 0xAA}});

  // New chunk fully contains an existing one: added around it.
  mem_cache.Clear();
  add(0x3040, 0x40, 0xAA);
  add(0x3000, 0x100, 0xBB);
  expect_l1({{0x3000, 0x40, 0xBB}, {0x3040, 0x40, 0xAA}, {0x3080, 0x80, 0xBB}});

  // New chunk is fully contained by an existing one: adds nothing.
  mem_cache.Clear();
  add(0x4000, 0x100, 0xAA);
  add(0x4080, 0x80, 0xBB);
  expect_l1({{0x4000, 0x100, 0xAA}});

  // New chunk partially overlaps two existing chunks; fills only the whole
  // between them.
  mem_cache.Clear();
  add(0x5000, 0x80, 0xAA);
  add(0x5100, 0x80, 0xCC);
  add(0x5040, 0x100, 0xBB);
  expect_l1({{0x5000, 0x80, 0xAA}, {0x5080, 0x80, 0xBB}, {0x5100, 0x80, 0xCC}});

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

  // Flush must erase an entry starting below the flushed address, and keep one
  // in the same line that it does not intersect.
  mem_cache.Clear();
  add(0x8100, 0x80, 0xAA);
  add(0x8000, 0x40, 0xBB);
  mem_cache.Flush(0x8140, 0x4);
  expect_l1({{0x8000, 0x40, 0xBB}});

  // A flush intersecting several partially overlapping chunks drops all of
  // them, while a chunk it does not intersect is left in place.
  mem_cache.Clear();
  add(0x9000, 0x80, 0xAA);
  add(0x9080, 0x40, 0xBB);
  add(0x9100, 0x80, 0xCC);
  mem_cache.Flush(0x9020, 0x80);
  expect_l1({{0x9100, 0x80, 0xCC}});
  expect_l2({});

  // Flush reaches a line and leaves the remainders on either side of it alone.
  mem_cache.Clear();
  add(0xB000 + line - 10, 10 + line + 10, 0xAA);
  expect_l2({{0xB000 + line, line, 0xAA}});
  mem_cache.Flush(0xB000 + line + 4, 0x4);
  expect_l1({{0xB000 + line - 10, 10, 0xAA}, {0xB000 + 2 * line, 10, 0xAA}});
  expect_l2({});

  // A whole line at an aligned address belongs to L2, not L1.
  mem_cache.Clear();
  add(0x9000, line, 0xAA);
  expect_l1({});
  expect_l2({{0x9000, line, 0xAA}});

  // A partial range at an aligned address stays in L1.
  mem_cache.Clear();
  add(0xA000, 0x40, 0xAA);
  expect_l1({{0xA000, 0x40, 0xAA}});
  expect_l2({});

  // An unaligned range longer than a line splits into one whole line plus a
  // remainder on each side.
  mem_cache.Clear();
  add(0xB000 + line - 10, 10 + line + 10, 0xAA);
  expect_l1({{0xB000 + line - 10, 10, 0xAA}, {0xB000 + 2 * line, 10, 0xAA}});
  expect_l2({{0xB000 + line, line, 0xAA}});

  // A range crossing a line boundary but covering no whole line splits in two.
  mem_cache.Clear();
  add(0xC000 + line - 12, 20, 0xAA);
  expect_l1({{0xC000 + line - 12, 12, 0xAA}, {0xC000 + line, 8, 0xAA}});
  expect_l2({});

  // A range already held by a line in L2 is dropped.
  mem_cache.Clear();
  add(0xD000, line, 0xAA);
  add(0xD000 + 8, 16, 0xBB);
  expect_l1({});
  expect_l2({{0xD000, line, 0xAA}});

  // A flush whose first line is absent from L2 still erases the later lines it
  // covers, because the range start is a lower bound and not a lookup.
  mem_cache.Clear();
  add(0xF000 + line, line, 0xAA);
  add(0xF000 + 2 * line, line, 0xBB);
  expect_l2({{0xF000 + line, line, 0xAA}, {0xF000 + 2 * line, line, 0xBB}});
  mem_cache.Flush(0xF000, 2 * line);
  expect_l2({{0xF000 + 2 * line, line, 0xBB}});

  // A whole line evicts an entry it only partly covers.
  mem_cache.Clear();
  add(0xE000 + line - 8, 16, 0xAA);
  add(0xE000 + line, line, 0xBB);
  expect_l1({{0xE000 + line - 8, 8, 0xAA}});
  expect_l2({{0xE000 + line, line, 0xBB}});
}

TEST_F(MemoryTest, TestReadStopsAtAnInvalidRange) {
  CacheTestProcess proc;
  ASSERT_TRUE(proc.GetProcess());
  DummyProcess *process = proc.GetProcess();
  MemoryCache &cache = process->GetMemoryCache();
  const lldb::addr_t base = 0xE000;

  cache.AddInvalidRange(base + 16, 16);
  process->SetMaxReadSize(4096);
  process->SetFiller(0xBB);
  process->m_reads.clear();

  // Only the bytes below the invalid range are served, and the read reports
  // the failure.
  Status error;
  std::vector<uint8_t> buf(64, 0);
  EXPECT_EQ(cache.Read(base, buf.data(), buf.size(), error), 16u);
  EXPECT_TRUE(error.Fail());
  EXPECT_TRUE(AllBytesAre(llvm::ArrayRef(buf).take_front(16), 0xBB));

  // The request stops where the invalid range starts, so the unreadable bytes
  // are never asked for.
  ASSERT_EQ(process->m_reads.size(), 1u);
  EXPECT_EQ(process->m_reads[0].first, base);
  EXPECT_EQ(process->m_reads[0].second, 16u);

  // A read starting inside the range has nothing to serve.
  Status inside_error;
  std::vector<uint8_t> inside(8, 0);
  process->m_reads.clear();
  EXPECT_EQ(cache.Read(base + 20, inside.data(), inside.size(), inside_error),
            0u);
  EXPECT_TRUE(inside_error.Fail());
  EXPECT_TRUE(process->m_reads.empty());
}

TEST_F(MemoryTest, TestReadRangesFromCaches) {
  CacheTestProcess proc;
  ASSERT_TRUE(proc.GetProcess());
  DummyProcess *process = proc.GetProcess();
  const uint64_t line_size = proc.GetLineSize();

  { // ReadRanges serves a range one entry covers.
    const lldb::addr_t base = 0x6000 + line_size - 10;
    process->GetMemoryCache().AddCacheData(
        base, std::make_shared<DataBufferHeap>(10 + line_size, 0xAA));
    process->SetMaxReadSize(0);
    llvm::SmallVector<uint8_t, 0> buffer(20, 0);
    llvm::SmallVector<Range<addr_t, size_t>> ranges = {{base, 20}};
    llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> results =
        process->ReadMemoryRanges(ranges, buffer);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].size(), 20u);
    EXPECT_TRUE(AllBytesAre(results[0], 0xAA));
  }

  { // An entry serves a range only if it covers it all.  A short fetch is all
    // the caller sees, and adds to L1 only the bytes no entry holds.
    TestMemoryCache cache(*process);
    AddCacheChunk(cache, 0xB000, 40, 0xAA);
    AddCacheChunk(cache, 0xC000, 8, 0xCC);
    process->SetMaxReadSize(20);
    process->SetFiller(0xBB);
    process->m_reads.clear();
    llvm::SmallVector<uint8_t, 0> buffer(72, 0);
    llvm::SmallVector<Range<addr_t, size_t>> ranges = {{0xB000, 64},
                                                       {0xC000, 8}};
    llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> results =
        cache.ReadRanges(ranges, buffer);
    ASSERT_EQ(results.size(), 2u);
    ASSERT_EQ(results[0].size(), 20u);
    EXPECT_TRUE(AllBytesAre(results[0], 0xBB));
    ASSERT_EQ(results[1].size(), 8u);
    EXPECT_TRUE(AllBytesAre(results[1], 0xCC));
    // A short reply is retried for the remainder, here with no bytes left.
    ASSERT_EQ(process->m_reads.size(), 2u);
    EXPECT_EQ(process->m_reads[0].first, 0xB000u);
    EXPECT_EQ(process->m_reads[0].second, 64u);
    EXPECT_EQ(process->m_reads[1].first, 0xB000u + 20u);
    EXPECT_EQ(process->m_reads[1].second, 64u - 20u);

    ASSERT_EQ(cache.GetL1Cache().GetSize(), 2u);
    const auto l1_chunks = Snapshot(cache.GetL1Cache());
    ASSERT_EQ(l1_chunks.size(), 2u);
    EXPECT_EQ(l1_chunks[0].first, 0xB000u);
    EXPECT_EQ(l1_chunks[1].first, 0xC000u);

    // The fetch's 20 bytes are all held already, so the chunk is unchanged.
    EXPECT_EQ(l1_chunks[0].second.size(), 40u);
    EXPECT_TRUE(AllBytesAre(l1_chunks[0].second, 0xAA));
    EXPECT_EQ(l1_chunks[1].second.size(), 8u);
    EXPECT_TRUE(AllBytesAre(l1_chunks[1].second, 0xCC));
  }

  { // A range ReadRanges fetched is cached, so asking for it again serves it
    // without going to the inferior.
    TestMemoryCache cache(*process);
    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xBB);
    process->m_reads.clear();
    llvm::SmallVector<uint8_t, 0> buffer(24, 0);
    llvm::SmallVector<Range<addr_t, size_t>> ranges = {{0x16000, 24}};
    llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> results =
        cache.ReadRanges(ranges, buffer);
    ASSERT_EQ(results.size(), 1u);
    ASSERT_EQ(results[0].size(), 24u);
    EXPECT_TRUE(AllBytesAre(results[0], 0xBB));
    ASSERT_FALSE(process->m_reads.empty());

    process->SetMaxReadSize(0);
    process->m_reads.clear();
    llvm::SmallVector<uint8_t, 0> again(24, 0);
    llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> second =
        cache.ReadRanges(ranges, again);
    ASSERT_EQ(second.size(), 1u);
    ASSERT_EQ(second[0].size(), 24u);
    EXPECT_TRUE(AllBytesAre(second[0], 0xBB));
    EXPECT_TRUE(process->m_reads.empty());
  }

  { // A range inside an invalid range gets an empty result without reaching the
    // inferior, and leaves the ranges on either side of it alone.
    TestMemoryCache cache(*process);
    AddCacheChunk(cache, 0x17000, 8, 0xAA);
    cache.AddInvalidRange(0x17100, 8);
    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xBB);
    process->m_reads.clear();
    llvm::SmallVector<uint8_t, 0> buffer(24, 0);
    llvm::SmallVector<Range<addr_t, size_t>> ranges = {
        {0x17000, 8}, {0x17100, 8}, {0x17200, 8}};
    llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> results =
        cache.ReadRanges(ranges, buffer);
    ASSERT_EQ(results.size(), 3u);
    ASSERT_EQ(results[0].size(), 8u); // a cache hit
    EXPECT_TRUE(AllBytesAre(results[0], 0xAA));
    EXPECT_TRUE(results[1].empty());  // the invalid range
    ASSERT_EQ(results[2].size(), 8u); // a miss, fetched
    EXPECT_TRUE(AllBytesAre(results[2], 0xBB));
    // Only the miss reached the inferior.
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, 0x17200u);
    EXPECT_EQ(process->m_reads[0].second, 8u);
  }
}

TEST_F(MemoryTest, TestReadRequestShape) {
  CacheTestProcess proc;
  ASSERT_TRUE(proc.GetProcess());
  DummyProcess *process = proc.GetProcess();
  const uint64_t line_size = proc.GetLineSize();

  { // The request starts at the cache entry, but longer than the cache entry.
    TestMemoryCache cache(*process);
    Status error;
    const lldb::addr_t base = 0x14000;

    //         v base           v base + line
    //   cache:|AAAAAAAAAAAAAAAA|
    // process:|BBBBBBBBBBBBBBBB|BBBBBBBBBBBBBBB|
    // buf    :|AAAAAAAAAAAAAAAA|BBB|
    //                              ^ base + line + 88
    AddCacheChunk(cache, base, 256, 0xAA);
    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xBB);
    process->m_reads.clear();
    std::vector<uint8_t> buf(line_size + 88, 0);
    EXPECT_EQ(cache.Read(base, buf.data(), buf.size(), error), buf.size());
    EXPECT_TRUE(AllBytesAre(llvm::ArrayRef(buf).take_front(256), 0xAA));
    EXPECT_TRUE(AllBytesAre(llvm::ArrayRef(buf).drop_front(256), 0xBB));
    // One request, from the first missing byte to the second line's end.
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, base + 256);
    EXPECT_EQ(process->m_reads[0].second, 2 * line_size - 256);
  }

  { // A read longer than a line that cache cannot serve, read the rest from the
    // inferior.
    //         v base         v base + line_size
    // cache:  |AAAAAAAAAAAAAA|AA|
    // process:|BBBBBBBBBBBBBB|BBBBBBBBBBBB|BBBBBBBBBBBB|
    // buf:    |AAAAAAAAAAAAAA|AABBBBBBBBBB|BBBBBBBBBBBB|
    TestMemoryCache cache(*process);
    Status error;
    const lldb::addr_t base = 0x15000;
    // A whole line plus a remainder.
    AddCacheChunk(cache, base, line_size + 8, 0xAA);
    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xBB);
    process->m_reads.clear();
    std::vector<uint8_t> buf(3 * line_size, 0);
    ASSERT_EQ(cache.Read(base, buf.data(), buf.size(), error), buf.size());
    EXPECT_TRUE(
        AllBytesAre(llvm::ArrayRef(buf).take_front(line_size + 8), 0xAA));
    EXPECT_TRUE(
        AllBytesAre(llvm::ArrayRef(buf).drop_front(line_size + 8), 0xBB));
    // One request, starting past the cached prefix and covering only the rest.
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, base + line_size + 8);
    EXPECT_EQ(process->m_reads[0].second, 2 * line_size - 8);

    // Cached where it was read from, so the same read now sends nothing and
    // returns the same bytes.
    process->SetMaxReadSize(0);
    process->m_reads.clear();
    std::vector<uint8_t> again(3 * line_size, 0);
    EXPECT_EQ(cache.Read(base, again.data(), again.size(), error),
              again.size());
    EXPECT_EQ(again, buf);
    EXPECT_TRUE(process->m_reads.empty());
  }

  { // Data split across L1 and L2 stitches back together.  The counting
    // pattern catches an offset error a uniform fill would hide.
    TestMemoryCache cache(*process);
    Status error;
    const lldb::addr_t base = 0x11000 + line_size - 7;
    const size_t size = 7 + 2 * line_size + 5;
    auto byte_at = [](size_t i) { return static_cast<uint8_t>(i * 7 + 1); };
    auto pattern = std::make_shared<DataBufferHeap>(size, 0);
    for (size_t i = 0; i < size; ++i)
      pattern->GetBytes()[i] = byte_at(i);
    cache.AddCacheData(base, pattern);

    // One remainder on each side and two whole lines between them.
    ASSERT_EQ(cache.GetL1Cache().GetSize(), 2u);
    ASSERT_EQ(cache.GetL2Cache().GetSize(), 2u);

    process->SetMaxReadSize(0);
    std::vector<uint8_t> buf(size, 0);
    ASSERT_EQ(cache.Read(base, buf.data(), buf.size(), error), size);
    for (size_t i = 0; i < size; ++i)
      ASSERT_EQ(buf[i], byte_at(i)) << "byte " << i;

    // A read starting inside an entry, which a read at its base cannot check.
    auto expect_at = [&](size_t offset, size_t len, const char *what) {
      SCOPED_TRACE(what);
      std::vector<uint8_t> got(len, 0);
      ASSERT_EQ(cache.Read(base + offset, got.data(), got.size(), error), len);
      for (size_t i = 0; i < len; ++i)
        ASSERT_EQ(got[i], byte_at(offset + i)) << "byte " << i;
    };

    expect_at(3, 4, "inside the leading L1 remainder");
    expect_at(7 + 9, 8, "inside the first whole line");
    expect_at(7 - 2, 8, "across the remainder into the line");
  }

  { // A short read must not hide cached bytes that start where it stopped.  The
    // count is taken from the caches, not from what the inferior returned.
    //       v base     v base+300
    // cache:           |CC|
    //   buf:|AAAAAAAAAAACC|
    //                     ^ base+310
    TestMemoryCache cache(*process);
    Status error;
    const lldb::addr_t base = 0x18000;

    AddCacheChunk(cache, base + 300, 10, 0xCC);
    process->SetMaxReadSize(300);
    process->SetFiller(0xAA);
    std::vector<uint8_t> buf(600, 0);
    EXPECT_EQ(cache.Read(base, buf.data(), buf.size(), error), 310u);
    EXPECT_TRUE(AllBytesAre(llvm::ArrayRef(buf).take_front(300), 0xAA));
    EXPECT_TRUE(AllBytesAre(llvm::ArrayRef(buf).slice(300, 10), 0xCC));
  }

  { // A read at an expedited chunk must not reach the inferior once a larger
    // read already covers it.
    //       v fp-line    v fp         v fp+line
    // cache:             |AAAA|
    //   buf:             |BBBBBBBBB|
    TestMemoryCache cache(*process);
    Status error;
    const lldb::addr_t fp = 0xF000 + line_size;
    cache.AddCacheData(fp, std::make_shared<DataBufferHeap>(16, 0xAA));

    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xBB);
    std::vector<uint8_t> big(2 * line_size, 0);
    ASSERT_EQ(cache.Read(fp - line_size, big.data(), big.size(), error),
              big.size());

    process->SetMaxReadSize(0);
    process->m_reads.clear();
    std::vector<uint8_t> out(32, 0);
    EXPECT_EQ(cache.Read(fp, out.data(), out.size(), error), out.size());
    EXPECT_TRUE(AllBytesAre(out, 0xBB));
    EXPECT_TRUE(process->m_reads.empty());
  }
}

// A read straddling two lines fetches whole lines, so both land in L2 and
// a later read of either hits.
TEST_F(MemoryTest, TestReadStraddlingTwoLines) {
  CacheTestProcess proc;
  ASSERT_TRUE(proc.GetProcess());
  DummyProcess *process = proc.GetProcess();
  const uint64_t line_size = proc.GetLineSize();
  const lldb::addr_t first = 0x20000;
  const lldb::addr_t second = first + line_size;

  { // Neither line cached: both are fetched whole, in one request.
    TestMemoryCache cache(*process);
    Status error;
    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xAA);
    process->m_reads.clear();

    std::vector<uint8_t> buf(16, 0);
    EXPECT_EQ(cache.Read(second - 8, buf.data(), buf.size(), error),
              buf.size());
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, first);
    EXPECT_EQ(process->m_reads[0].second, 2 * line_size);
    EXPECT_TRUE(cache.GetL2Cache().Holds(first));
    EXPECT_TRUE(cache.GetL2Cache().Holds(second));

    process->m_reads.clear();
    std::vector<uint8_t> again(8, 0);
    EXPECT_EQ(cache.Read(first, again.data(), again.size(), error),
              again.size());
    EXPECT_TRUE(process->m_reads.empty());
  }

  { // The second line is already in L2, only the first is missing, so only
    // that one is fetched and the second is not sent again.
    TestMemoryCache cache(*process);
    Status error;
    std::vector<uint8_t> whole(line_size, 0xBB);
    cache.AddCacheData(second, whole.data(), whole.size());
    ASSERT_TRUE(cache.GetL2Cache().Holds(second));

    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xAA);
    process->m_reads.clear();
    std::vector<uint8_t> buf(16, 0);
    EXPECT_EQ(cache.Read(second - 8, buf.data(), buf.size(), error),
              buf.size());
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, first);
    EXPECT_EQ(process->m_reads[0].second, line_size);
    // The tail of the request came out of the line that was already there.
    EXPECT_TRUE(AllBytesAre(llvm::ArrayRef(buf).take_front(8), 0xAA));
    EXPECT_TRUE(AllBytesAre(llvm::ArrayRef(buf).drop_front(8), 0xBB));
  }

  { // A short second line does not count as present: skipping it would leave a
    // hole, so both lines are fetched.
    TestMemoryCache cache(*process);
    Status error;
    std::vector<uint8_t> partial(line_size / 2, 0xBB);
    cache.AddCacheData(second, partial.data(), partial.size());

    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xAA);
    process->m_reads.clear();
    std::vector<uint8_t> buf(16, 0);
    EXPECT_EQ(cache.Read(second - 8, buf.data(), buf.size(), error),
              buf.size());
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].second, 2 * line_size);
    EXPECT_EQ(cache.GetL2Cache().GetSize(), 2u);
    EXPECT_EQ(cache.GetL1Cache().GetSize(), 0u);
  }

  { // A request longer than a line still grows when it fits in the two lines it
    // touches.
    TestMemoryCache cache(*process);
    Status error;
    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xAA);
    process->m_reads.clear();

    std::vector<uint8_t> buf(line_size + 4, 0);
    EXPECT_EQ(cache.Read(first, buf.data(), buf.size(), error), buf.size());
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, first);
    EXPECT_EQ(process->m_reads[0].second, 2 * line_size);
    EXPECT_TRUE(cache.GetL2Cache().Holds(first));
    EXPECT_TRUE(cache.GetL2Cache().Holds(second));

    process->m_reads.clear();
    std::vector<uint8_t> again(8, 0);
    EXPECT_EQ(
        cache.Read(second + line_size - 8, again.data(), again.size(), error),
        again.size());
    EXPECT_TRUE(process->m_reads.empty());
  }

  { // A request reaching a third line is read as asked.
    //        v first      v first+line v fist+2*line
    //  cache:|            |            |
    //    buf:  |            |            |
    //          ^ first+8                 ^first+8+2*line
    TestMemoryCache cache(*process);
    Status error;
    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xAA);
    process->m_reads.clear();

    std::vector<uint8_t> buf(2 * line_size, 0);
    EXPECT_EQ(cache.Read(first + 8, buf.data(), buf.size(), error), buf.size());
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, first + 8);
    EXPECT_EQ(process->m_reads[0].second, 2 * line_size);
  }

  {
    TestMemoryCache cache(*process);
    Status error;
    std::vector<uint8_t> held(8, 0xCC);
    cache.AddCacheData(first, held.data(), held.size());

    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xAA);
    process->m_reads.clear();
    std::vector<uint8_t> buf(line_size + 4, 0);
    EXPECT_EQ(cache.Read(first, buf.data(), buf.size(), error), buf.size());
    ASSERT_EQ(process->m_reads.size(), 1u);
    // A prefix served from the caches stops the straddle arm growing down.
    EXPECT_EQ(process->m_reads[0].first, first + 8);
    EXPECT_TRUE(AllBytesAre(llvm::ArrayRef(buf).take_front(8), 0xCC));
  }
}

TEST_F(MemoryTest, TestReadGrowthAgainstInvalidRanges) {
  CacheTestProcess proc;
  ASSERT_TRUE(proc.GetProcess());
  DummyProcess *process = proc.GetProcess();
  const uint64_t line_size = proc.GetLineSize();
  const lldb::addr_t first = 0x30000;
  const lldb::addr_t second = first + line_size;

  { // An invalid range below the read, ending inside the line.
    TestMemoryCache cache(*process);
    Status error;
    cache.AddInvalidRange(first, 0x100);
    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xAA);
    process->m_reads.clear();

    std::vector<uint8_t> buf(16, 0);
    EXPECT_EQ(cache.Read(first + 0x180, buf.data(), buf.size(), error),
              buf.size());
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, first + 0x180);
    EXPECT_EQ(process->m_reads[0].second, line_size - 0x180);
  }

  { // An invalid range in the bytes growth adds above, inside the same line.
    TestMemoryCache cache(*process);
    Status error;
    cache.AddInvalidRange(first + 0x180, 0x80);
    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xAA);
    process->m_reads.clear();

    std::vector<uint8_t> buf(16, 0);
    EXPECT_EQ(cache.Read(first + 0x100, buf.data(), buf.size(), error),
              buf.size());
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, first);
    EXPECT_EQ(process->m_reads[0].second, 0x180u);
  }

  { // An invalid range in the second line, which growth would add: the request
    // stops below it.
    TestMemoryCache cache(*process);
    Status error;
    cache.AddInvalidRange(second + 0x100, 0x80);
    process->SetMaxReadSize(4 * line_size);
    process->SetFiller(0xAA);
    process->m_reads.clear();

    std::vector<uint8_t> buf(16, 0);
    EXPECT_EQ(cache.Read(second - 8, buf.data(), buf.size(), error),
              buf.size());
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, first);
    EXPECT_EQ(process->m_reads[0].second, line_size + 0x100);
  }
}

// A flushed range whose end wraps past UINT64_MAX must stop at the top line.
TEST_F(MemoryTest, TestFlushAtTheTopOfTheAddressSpace) {
  CacheTestProcess proc;
  ASSERT_TRUE(proc.GetProcess());
  DummyProcess *process = proc.GetProcess();
  const uint64_t line_size = proc.GetLineSize();
  const lldb::addr_t top_line = UINT64_MAX - line_size + 1;
  const lldb::addr_t line_below_top = top_line - line_size;

  TestMemoryCache cache(*process);
  Status error;
  process->SetMaxReadSize(4 * line_size);
  std::vector<uint8_t> buf(line_size, 0);
  cache.Read(top_line, buf.data(), buf.size(), error);
  cache.Read(0, buf.data(), buf.size(), error);
  ASSERT_EQ(cache.GetL2Cache().GetSize(), 2u);
  ASSERT_TRUE(cache.GetL2Cache().Holds(top_line));

  // This range ends past UINT64_MAX.
  cache.Flush(UINT64_MAX - 8, 100);
  EXPECT_FALSE(cache.GetL2Cache().Holds(top_line));
  EXPECT_TRUE(cache.GetL2Cache().Holds(0));

  { // A one-byte flush of the last byte covers only the topmost line, so it
    // must not reach the line below it.
    TestMemoryCache cache(*process);
    cache.AddCacheData(line_below_top, buf.data(), buf.size());
    cache.AddCacheData(top_line, buf.data(), buf.size());
    ASSERT_EQ(cache.GetL2Cache().GetSize(), 2u);

    cache.Flush(UINT64_MAX, 1);
    EXPECT_FALSE(cache.GetL2Cache().Holds(top_line));
    EXPECT_TRUE(cache.GetL2Cache().Holds(line_below_top));
  }
}

// The cache copies raw bytes, which have no buffer behind them to retain.
TEST_F(MemoryTest, TestCacheCopiesRawBytes) {
  CacheTestProcess proc;
  ASSERT_TRUE(proc.GetProcess());
  DummyProcess *process = proc.GetProcess();
  Status error;
  TestMemoryCache cache(*process);
  std::vector<uint8_t> raw(16, 0xAA);
  cache.AddCacheData(0x5000, raw.data(), raw.size());
  ASSERT_TRUE(cache.GetL1Cache().Holds(0x5000));
  EXPECT_NE(cache.GetL1Cache().Lookup(0x5000).data(), raw.data());

  // Editing the caller's bytes must not change what the cache returns, and the
  // inferior supplies nothing, so every byte read came from the cache.
  raw.assign(raw.size(), 0xBB);
  process->SetMaxReadSize(0);
  std::vector<uint8_t> out(16, 0);
  EXPECT_EQ(cache.Read(0x5000, out.data(), out.size(), error), out.size());
  ASSERT_TRUE(process->m_reads.empty());
  EXPECT_TRUE(AllBytesAre(out, 0xAA));
}

TEST_F(MemoryTest, TestUnusableCacheLineSize) {
  ArchSpec arch("arm64-apple-macosx");

  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  // A Process copies the global properties when it is constructed, so the
  // setting must be in place before CreateProcess, and put back afterwards.
  struct SettingGuard {
    ~SettingGuard() {
      Process::GetGlobalProperties().SetPropertyValue(
          nullptr, eVarSetOperationClear, "memory-cache-line-size", "");
    }
  } restore_setting;

  auto set_line_size = [](const char *setting) {
    return Process::GetGlobalProperties().SetPropertyValue(
        nullptr, eVarSetOperationAssign, "memory-cache-line-size", setting);
  };

  // A usable setting must take effect, or the checks below prove nothing.
  ASSERT_TRUE(set_line_size("256").Success());
  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  DummyProcess *process =
      static_cast<DummyProcess *>(CreateProcess(target_sp).get());
  EXPECT_EQ(process->GetMemoryCacheLineSize(), 256u);

  for (const char *setting : {"0", "4294967296"}) {
    SCOPED_TRACE(setting);
    EXPECT_TRUE(set_line_size(setting).Fail());
    // Refused, so the last usable value is still in effect.
    EXPECT_EQ(process->GetMemoryCacheLineSize(), 256u);
    TargetSP later_target_sp = CreateTarget(debugger_sp, arch);
    EXPECT_EQ(CreateProcess(later_target_sp)->GetMemoryCacheLineSize(), 256u);
  }
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

  size_t DoReadMemory(const ProcessAddress &process_addr, void *buf,
                      size_t size, Status &error) override {
    lldb::addr_t vm_addr = process_addr.GetValue();
    if (read_less_than_requested && size > 0)
      size--;
    if (read_more_than_requested)
      size *= 2;
    uint8_t *buffer = static_cast<uint8_t *>(buf);
    for (lldb::addr_t addr = vm_addr; addr < vm_addr + size; addr++)
      buffer[addr - vm_addr] = static_cast<uint8_t>(addr); // LSB of addr.
    return size;
  }
  MemoryCache &GetMemoryCache() { return m_memory_cache; }
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

TEST_F(MemoryTest, TestReadMemoryRangesUsesL2Cache) {
  ArchSpec arch("x86_64-apple-macosx-");

  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);

  ProcessSP process_sp = CreateProcess(target_sp);
  ASSERT_TRUE(process_sp);

  DummyProcess *process = static_cast<DummyProcess *>(process_sp.get());
  const uint64_t l2_cache_size = process->GetMemoryCacheLineSize();
  Status error;
  uint8_t header[8];

  // Read the first 8 bytes of a cache line, the way a caller reads the header
  // of an array before batching the elements that follow it. This fills the
  // whole line and leaves the inferior unable to supply anything more.
  const addr_t full_line = 0x1000;
  ASSERT_EQ(full_line % l2_cache_size, 0u);
  process->SetMaxReadSize(l2_cache_size);
  process->SetFiller('A');
  process->m_reads.clear();
  ASSERT_EQ(process->ReadMemory(full_line, header, sizeof(header), error),
            sizeof(header));
  ASSERT_EQ(process->m_reads.size(), 1u);
  EXPECT_EQ(process->m_reads[0].first, full_line);
  EXPECT_EQ(process->m_reads[0].second, l2_cache_size);

  { // Ranges covered by that line are served from the cache. Leave the inferior
    // able to answer, with a filler of its own, so a miss would show up both in
    // the contents below and in the request log.
    process->SetMaxReadSize(l2_cache_size);
    process->SetFiller('X');
    process->m_reads.clear();
    llvm::SmallVector<uint8_t, 0> buffer(3 * 8, 0);
    llvm::SmallVector<Range<addr_t, size_t>> ranges = {
        {full_line + 8, 8},
        {full_line + 16, 8},
        {full_line + l2_cache_size - 8, 8}};
    llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> read_results =
        process->ReadMemoryRanges(ranges, buffer);
    ASSERT_EQ(read_results.size(), ranges.size());
    for (llvm::MutableArrayRef<uint8_t> memory : read_results) {
      ASSERT_EQ(memory.size(), 8u);
      for (uint8_t byte : memory)
        EXPECT_EQ(byte, 'A');
    }
    // Nothing was read from the inferior, so no packet was sent.
    EXPECT_TRUE(process->m_reads.empty());
  }

  { // A range crossing into the next, uncached line is a miss.
    process->SetMaxReadSize(0);
    process->m_reads.clear();
    llvm::SmallVector<uint8_t, 0> buffer(8, 0);
    llvm::SmallVector<Range<addr_t, size_t>> ranges = {
        {full_line + l2_cache_size - 4, 8}};
    llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> read_results =
        process->ReadMemoryRanges(ranges, buffer);
    ASSERT_EQ(read_results.size(), 1u);
    EXPECT_EQ(read_results[0].size(), 0u);
    // The missed range is asked for as it stands, not rounded up to a line.
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, full_line + l2_cache_size - 4);
    EXPECT_EQ(process->m_reads[0].second, 8u);
  }

  { // A batch of hits and misses keeps the results in the requested order, and
    // asks the inferior for the missed range only.
    const addr_t uncached_line = 0x3000;
    process->SetMaxReadSize(l2_cache_size);
    process->SetFiller('C');
    process->m_reads.clear();
    llvm::SmallVector<uint8_t, 0> buffer(3 * 8, 0);
    llvm::SmallVector<Range<addr_t, size_t>> ranges = {
        {full_line + 8, 8}, {uncached_line, 8}, {full_line + 16, 8}};
    llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> read_results =
        process->ReadMemoryRanges(ranges, buffer);
    ASSERT_EQ(read_results.size(), ranges.size());
    for (llvm::MutableArrayRef<uint8_t> memory : read_results)
      ASSERT_EQ(memory.size(), 8u);
    for (uint8_t byte : read_results[0])
      EXPECT_EQ(byte, 'A');
    for (uint8_t byte : read_results[1])
      EXPECT_EQ(byte, 'C');
    for (uint8_t byte : read_results[2])
      EXPECT_EQ(byte, 'A');
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, uncached_line);
    EXPECT_EQ(process->m_reads[0].second, 8u);
  }

  // A line the inferior could only partially supply is cached short.
  const addr_t short_line = 0x2000;
  ASSERT_EQ(short_line % l2_cache_size, 0u);
  const size_t bytes_available = 64;
  ASSERT_LT(bytes_available, l2_cache_size);
  process->SetMaxReadSize(bytes_available);
  process->SetFiller('D');
  process->m_reads.clear();
  ASSERT_EQ(process->ReadMemory(short_line, header, sizeof(header), error),
            sizeof(header));
  ASSERT_EQ(process->m_reads.size(), 2u);
  EXPECT_EQ(process->m_reads[0].first, short_line);
  EXPECT_EQ(process->m_reads[0].second, l2_cache_size);
  EXPECT_EQ(process->m_reads[1].first, short_line + bytes_available);
  EXPECT_EQ(process->m_reads[1].second, l2_cache_size - bytes_available);

  { // Only the part of the line that was actually read may be served.
    process->m_reads.clear();
    llvm::SmallVector<uint8_t, 0> buffer(2 * 8, 0);
    llvm::SmallVector<Range<addr_t, size_t>> ranges = {
        {short_line + bytes_available - 8, 8},
        {short_line + bytes_available - 4, 8}};
    llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> read_results =
        process->ReadMemoryRanges(ranges, buffer);
    ASSERT_EQ(read_results.size(), ranges.size());
    ASSERT_EQ(read_results[0].size(), 8u);
    for (uint8_t byte : read_results[0])
      EXPECT_EQ(byte, 'D');
    EXPECT_EQ(read_results[1].size(), 0u);
    ASSERT_EQ(process->m_reads.size(), 1u);
    EXPECT_EQ(process->m_reads[0].first, short_line + bytes_available - 4);
    EXPECT_EQ(process->m_reads[0].second, 8u);
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

TEST_F(MemoryDeathTest, TestReadRangesWithShortBufferAndCacheHit) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  ArchSpec arch("arm64-apple-macosx");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ASSERT_TRUE(target_sp);
  ProcessSP process_sp = CreateProcess(target_sp);
  ASSERT_TRUE(process_sp);

  DummyProcess *process = static_cast<DummyProcess *>(process_sp.get());
  TestMemoryCache cache(*process);
  cache.AddCacheData(0x1000, std::make_shared<DataBufferHeap>(16, 0xAA));
  ASSERT_TRUE(cache.GetL1Cache().Holds(0x1000));

  llvm::SmallVector<uint8_t, 0> short_buffer(8, 0);
  llvm::SmallVector<Range<addr_t, size_t>> ranges = {{0x1000, 16}};
  llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> read_results;
  ASSERT_DEBUG_DEATH(
      { read_results = cache.ReadRanges(ranges, short_buffer); },
      "MemoryCache::ReadRanges: provided buffer is too short");
#ifdef NDEBUG
  // With asserts off, the ranges come back empty instead.
  ASSERT_EQ(read_results.size(), ranges.size());
  for (llvm::MutableArrayRef<uint8_t> result : read_results)
    ASSERT_TRUE(result.empty());
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

  // Memory cache has to be off to reach the one in Process::DoReadMemoryRanges.
  Status set_error = process_sp->SetPropertyValue(
      nullptr, eVarSetOperationAssign, "disable-memory-cache", "true");
  ASSERT_TRUE(set_error.Success()) << set_error.AsCString();
  ASSERT_TRUE(process_sp->GetDisableMemoryCache());

  llvm::SmallVector<uint8_t, 0> short_buffer(10, 0);
  llvm::SmallVector<Range<addr_t, size_t>> ranges = {{0x12345, 128},
                                                     {0x11, 128}};
  llvm::SmallVector<llvm::MutableArrayRef<uint8_t>> read_results;
  ASSERT_DEBUG_DEATH(
      { read_results = process_sp->ReadMemoryRanges(ranges, short_buffer); },
      "Process::DoReadMemoryRanges: provided buffer is too short");
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

  size_t DoReadMemory(const ProcessAddress &process_addr, void *buf,
                      size_t size, Status &error) override {
    lldb::addr_t vm_addr = process_addr.GetValue();
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

#ifndef NDEBUG
TEST_F(MemoryDeathTest, TestVerifyMemoryReads) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  ArchSpec arch("x86_64-apple-macosx-");
  Platform::SetHostPlatform(PlatformRemoteMacOSX::CreateInstance(true, &arch));
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  TargetSP target_sp = CreateTarget(debugger_sp, arch);
  ListenerSP listener_sp(Listener::MakeListener("dummy"));
  auto process_sp =
      std::make_shared<DummyReaderProcess>(target_sp, listener_sp);

  // Off by default, and set on this process, so there is nothing to restore.
  ASSERT_FALSE(process_sp->GetVerifyMemoryReads());
  Status set_error = process_sp->SetPropertyValue(
      nullptr, eVarSetOperationAssign, "verify-memory-reads", "true");
  ASSERT_TRUE(set_error.Success()) << set_error.AsCString();
  ASSERT_TRUE(process_sp->GetVerifyMemoryReads());

  // A cache that agrees with the process passes, and still returns the bytes.
  Status error;
  std::vector<uint8_t> buf(16, 0);
  EXPECT_EQ(process_sp->ReadMemory(0x1000, buf.data(), buf.size(), error),
            buf.size());
  for (size_t i = 0; i < buf.size(); ++i)
    ASSERT_EQ(buf[i], static_cast<uint8_t>(0x1000 + i)) << "byte " << i;

  // The same holds for the ranges API.
  llvm::SmallVector<Range<addr_t, size_t>> ranges = {{0x1000, 16},
                                                     {0x3000, 16}};
  llvm::SmallVector<uint8_t, 0> ranges_buf(32, 0);
  for (auto [range, memory] :
       llvm::zip(ranges, process_sp->ReadMemoryRanges(ranges, ranges_buf))) {
    ASSERT_EQ(memory.size(), 16u);
    for (auto [i, byte] : llvm::enumerate(memory))
      ASSERT_EQ(byte, static_cast<uint8_t>(range.GetRangeBase() + i));
  }

  // DummyReaderProcess returns the low byte of each address, so a run of
  // zeroes cannot be what it would read.
  process_sp->GetMemoryCache().Clear();
  process_sp->GetMemoryCache().AddCacheData(
      0x2000, std::make_shared<DataBufferHeap>(16, 0));
  std::vector<uint8_t> bad(16, 0);
  ASSERT_DEATH(
      { process_sp->ReadMemory(0x2000, bad.data(), bad.size(), error); },
      "memory cache returned something the process did not");
  Range<addr_t, size_t> bad_range(0x2000, 16);
  ASSERT_DEATH(
      { process_sp->ReadMemoryRanges(bad_range, bad); },
      "memory cache returned something the process did not");
}
#endif // NDEBUG

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
  size_t DoReadMemory(const ProcessAddress &, void *, size_t,
                      Status &) override {
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
