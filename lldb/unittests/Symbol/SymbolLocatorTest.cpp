//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/SymbolLocator.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Utility/FileSpecList.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

using namespace lldb;
using namespace lldb_private;

namespace {

/// Which steps of the search ran, so a test can tell where an answer came from.
/// Written from every thread of a batch, so the flags have to be atomic.
struct LocatorCalls {
  std::atomic<bool> located_symbol_file = false;
  std::atomic<bool> located_object_file = false;
  std::atomic<bool> downloaded = false;

  void Clear() {
    located_symbol_file = false;
    located_object_file = false;
    downloaded = false;
  }
};

LocatorCalls g_calls;

/// What the object file locator claims to have found, if anything.
std::optional<FileSpec> g_object_file;

/// What the symbol file locator claims to have found, if anything.
std::optional<FileSpec> g_symbol_file;

/// What the symbol server claims to have downloaded, if anything.
std::optional<FileSpec> g_downloaded_symbol_file;

/// When set, the symbol server fails the way a failure to launch it does, with
/// an errno rather than a message.
bool g_symbol_server_errno = false;

/// When set, the fake locator only answers for requests carrying a UUID.
bool g_only_with_uuid = false;

/// Run by the fake locator, to let a test hold every search of a batch open at
/// once.
std::function<void()> g_barrier;

std::optional<FileSpec> LocateExecutableSymbolFile(const ModuleSpec &,
                                                   const FileSpecList &) {
  g_calls.located_symbol_file = true;
  return g_symbol_file;
}

std::optional<ModuleSpec> LocateExecutableObjectFile(const ModuleSpec &spec) {
  g_calls.located_object_file = true;
  if (g_barrier)
    g_barrier();
  if (!g_object_file)
    return {};
  if (g_only_with_uuid && !spec.GetUUID().IsValid())
    return {};
  ModuleSpec located(spec);
  located.GetFileSpec() = *g_object_file;
  return located;
}

bool DownloadObjectAndSymbolFile(ModuleSpec &module_spec, Status &error,
                                 bool force_lookup, bool) {
  g_calls.downloaded = true;
  if (!force_lookup)
    return false;
  if (g_symbol_server_errno)
    error = Status(std::make_error_code(std::errc::too_many_files_open));
  else
    error = Status::FromErrorString("the symbol server said no");
  // A server that delivers is under no obligation to leave the Status alone.
  if (!g_downloaded_symbol_file)
    return false;
  module_spec.GetSymbolFileSpec() = *g_downloaded_symbol_file;
  return true;
}

SymbolLocator *CreateSymbolLocator() { return nullptr; }

/// A platform that answers out of an index of its own, the way
/// PlatformDarwinKernel answers for kexts.
class IndexedPlatform : public Platform {
public:
  IndexedPlatform() : Platform(/*is_host_platform=*/false) {}

  llvm::StringRef GetPluginName() override { return "indexed"; }
  llvm::StringRef GetDescription() override { return "test platform"; }
  std::vector<ArchSpec> GetSupportedArchitectures(const ArchSpec &) override {
    return {};
  }
  lldb::ProcessSP Attach(ProcessAttachInfo &, Debugger &, Target *,
                         Status &) override {
    return nullptr;
  }
  void CalculateTrapHandlerSymbolNames() override {}
  UserIDResolver &GetUserIDResolver() override {
    return UserIDResolver::GetNoopResolver();
  }

  std::optional<ModuleSpec> FindModuleFiles(const ModuleSpec &spec,
                                            const FileSpecList &,
                                            StatisticsMap &) override {
    ++find_module_files_calls;
    if (!m_answer)
      return {};
    ModuleSpec found(spec);
    found.GetFileSpec() = *m_answer;
    return found;
  }

  std::optional<FileSpec> m_answer;
  unsigned find_module_files_calls = 0;
};

class SymbolLocatorTest : public testing::Test {
public:
  SymbolLocatorTest()
      : m_empty_buffer("", "<empty buffer>"),
        m_fs(new llvm::vfs::InMemoryFileSystem()) {}

  void SetUp() override {
    // The batch runs on the debugger's thread pool. Debugger::Initialize takes
    // an argument, so SubsystemRAII cannot call it.
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });

    // Locate reports a binary it cannot find as an error, so a test that wants
    // a hit has to point the fake locator at a file that exists.
    FileSystem::Initialize(m_fs);
    HostInfo::Initialize();
    m_fs->addFileNoOwn(m_binary.GetPath(), 0, m_empty_buffer);
    m_fs->addFileNoOwn(m_symbols.GetPath(), 0, m_empty_buffer);

    g_calls.Clear();
    g_object_file = std::nullopt;
    g_symbol_file = std::nullopt;
    g_downloaded_symbol_file = std::nullopt;
    g_symbol_server_errno = false;
    g_only_with_uuid = false;
    g_barrier = nullptr;
    ASSERT_TRUE(PluginManager::RegisterPlugin(
        "test", "test symbol locator", CreateSymbolLocator,
        LocateExecutableObjectFile, LocateExecutableSymbolFile,
        DownloadObjectAndSymbolFile));
  }

  void TearDown() override {
    g_barrier = nullptr;
    PluginManager::UnregisterPlugin(CreateSymbolLocator);
    HostInfo::Terminate();
    FileSystem::Terminate();
  }

  llvm::MemoryBufferRef m_empty_buffer;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> m_fs;

  /// Files that exist, for the cases that want a hit.
  FileSpec m_binary = FileSpec("/binary", FileSpec::Style::posix);
  FileSpec m_symbols = FileSpec("/binary.dSYM", FileSpec::Style::posix);
};

std::vector<SymbolLocator::Request> MakeRequests(size_t count) {
  return std::vector<SymbolLocator::Request>(count);
}

} // namespace

TEST_F(SymbolLocatorTest, MissRunsEveryStep) {
  llvm::Expected<SymbolLocator::Result> result =
      SymbolLocator::Locate(SymbolLocator::Request(), FileSpecList());

  EXPECT_TRUE(g_calls.located_symbol_file);
  EXPECT_TRUE(g_calls.located_object_file);
  // Neither file was found, so the symbol server is the last resort.
  EXPECT_TRUE(g_calls.downloaded);

  // Nothing was found and no symbol server was asked, so there is nothing to
  // say beyond the fact of the miss. The caller composes its own message.
  ASSERT_FALSE(result);
  llvm::Error error = result.takeError();
  EXPECT_TRUE(error.isA<SymbolLocator::NotFound>());
  llvm::consumeError(std::move(error));
}

TEST_F(SymbolLocatorTest, ReportsWhatThePluginsFound) {
  g_object_file = m_binary;
  g_symbol_file = m_symbols;

  llvm::Expected<SymbolLocator::Result> result =
      SymbolLocator::Locate(SymbolLocator::Request(), FileSpecList());

  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  EXPECT_EQ(m_binary, result->module_spec.GetFileSpec());
  EXPECT_EQ(m_symbols, result->module_spec.GetSymbolFileSpec());
  // Both files were in hand, so there was nothing to ask a symbol server for.
  EXPECT_FALSE(g_calls.downloaded);
}

TEST_F(SymbolLocatorTest, MissCarriesSymbolServerError) {
  SymbolLocator::Request request;
  request.external_lookup = true;

  llvm::Expected<SymbolLocator::Result> result =
      SymbolLocator::Locate(request, FileSpecList());

  ASSERT_FALSE(result);
  llvm::Error error = result.takeError();
  EXPECT_FALSE(error.isA<SymbolLocator::NotFound>());
  EXPECT_EQ("the symbol server said no", llvm::toString(std::move(error)));
}

TEST_F(SymbolLocatorTest, BinaryWithoutSymbolsIsASuccess) {
  // A binary without symbols is still worth loading, and the caller still gets
  // to tell the user why the symbols are missing.
  g_object_file = m_binary;
  SymbolLocator::Request request;
  request.external_lookup = true;

  llvm::Expected<SymbolLocator::Result> result =
      SymbolLocator::Locate(request, FileSpecList());

  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  EXPECT_EQ(m_binary, result->module_spec.GetFileSpec());
  EXPECT_FALSE(result->module_spec.GetSymbolFileSpec());
  ASSERT_TRUE(result->symbol_error);
  EXPECT_EQ("the symbol server said no",
            llvm::toString(std::move(*result->symbol_error)));
}

TEST_F(SymbolLocatorTest, ASuccessfulDownloadIsNotASymbolError) {
  // A symbol server that delivered has nothing left to explain.
  g_object_file = m_binary;
  g_downloaded_symbol_file = m_symbols;
  SymbolLocator::Request request;
  request.external_lookup = true;

  llvm::Expected<SymbolLocator::Result> result =
      SymbolLocator::Locate(request, FileSpecList());

  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  EXPECT_EQ(m_symbols, result->module_spec.GetSymbolFileSpec());
  // Consumed in the failure message, because an unchecked Error aborts when it
  // goes out of scope.
  EXPECT_FALSE(result->symbol_error)
      << llvm::toString(std::move(*result->symbol_error));
}

TEST_F(SymbolLocatorTest, SymbolFileWithoutBinaryIsAMiss) {
  // Symbols with no binary to apply them to are of no use, so this is the hard
  // failure and not a success carrying half an answer.
  g_symbol_file = m_symbols;

  llvm::Expected<SymbolLocator::Result> result =
      SymbolLocator::Locate(SymbolLocator::Request(), FileSpecList());

  EXPECT_THAT_EXPECTED(result, llvm::Failed());
}

TEST_F(SymbolLocatorTest, AnErrnoFromTheSymbolServerIsNotAPlainMiss) {
  // A Status carrying an errno converts to an llvm::ECError, so an error code
  // is not enough to tell a real failure from the miss sentinel. Failing to
  // even launch a symbol server has to reach the user.
  SymbolLocator::Request request;
  request.external_lookup = true;
  g_symbol_server_errno = true;

  llvm::Expected<SymbolLocator::Result> result =
      SymbolLocator::Locate(request, FileSpecList());

  ASSERT_FALSE(result);
  llvm::Error error = result.takeError();
  EXPECT_FALSE(error.isA<SymbolLocator::NotFound>());
  llvm::consumeError(std::move(error));
}

TEST_F(SymbolLocatorTest, ThePlatformAnswersBeforeThePlugins) {
  auto platform = std::make_shared<IndexedPlatform>();
  platform->m_answer = m_binary;

  SymbolLocator::Request request;
  request.platform = platform;

  llvm::Expected<SymbolLocator::Result> result =
      SymbolLocator::Locate(request, FileSpecList());

  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  EXPECT_EQ(1u, platform->find_module_files_calls);
  EXPECT_EQ(m_binary, result->module_spec.GetFileSpec());
  EXPECT_FALSE(g_calls.located_symbol_file);
  EXPECT_FALSE(g_calls.located_object_file);
  EXPECT_FALSE(g_calls.downloaded);
}

TEST_F(SymbolLocatorTest, ThePluginsRunWhenThePlatformHasNothingToSay) {
  auto platform = std::make_shared<IndexedPlatform>();
  platform->m_answer = std::nullopt;
  g_object_file = m_binary;

  SymbolLocator::Request request;
  request.platform = platform;

  llvm::Expected<SymbolLocator::Result> result =
      SymbolLocator::Locate(request, FileSpecList());

  ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
  EXPECT_EQ(1u, platform->find_module_files_calls);
  EXPECT_TRUE(g_calls.located_object_file);
}

TEST_F(SymbolLocatorTest, TheBatchSearchesEveryRequest) {
  g_object_file = m_binary;
  std::vector<SymbolLocator::Request> requests = MakeRequests(8);

  std::vector<llvm::Expected<SymbolLocator::Result>> results =
      SymbolLocator::Locate(requests, FileSpecList(), /*parallel=*/true);

  ASSERT_EQ(requests.size(), results.size());
  for (llvm::Expected<SymbolLocator::Result> &result : results) {
    ASSERT_THAT_EXPECTED(result, llvm::Succeeded());
    EXPECT_EQ(m_binary, result->module_spec.GetFileSpec());
  }
}

TEST_F(SymbolLocatorTest, TheBatchKeepsResultsInRequestOrder) {
  // Every other request is one the locator will answer, so the results can only
  // line up with the requests if the order is kept.
  g_object_file = m_binary;
  g_only_with_uuid = true;
  std::vector<SymbolLocator::Request> requests = MakeRequests(6);
  for (auto [i, request] : llvm::enumerate(requests))
    if (i % 2 == 0)
      request.module_spec.GetUUID() = UUID("0123456789ABCDEF", 16);

  std::vector<llvm::Expected<SymbolLocator::Result>> results =
      SymbolLocator::Locate(requests, FileSpecList(), /*parallel=*/true);

  ASSERT_EQ(requests.size(), results.size());
  for (auto [i, result] : llvm::enumerate(results)) {
    if (i % 2 == 0)
      EXPECT_THAT_EXPECTED(result, llvm::Succeeded()) << "request " << i;
    else
      EXPECT_THAT_EXPECTED(result, llvm::Failed()) << "request " << i;
  }
}

TEST_F(SymbolLocatorTest, TheBatchSearchesConcurrently) {
  // A serial batch could never get every task inside the locator at once. No
  // more tasks than the pool can run, or the ones left queued would hang it.
  const size_t num_requests =
      std::min<size_t>(4, Debugger::GetThreadPool().getMaxConcurrency());
  if (num_requests < 2)
    GTEST_SKIP() << "the thread pool runs one task at a time";

  std::mutex mutex;
  std::condition_variable cv;
  size_t arrived = 0;
  bool everyone_arrived = false;

  g_barrier = [&] {
    std::unique_lock<std::mutex> lock(mutex);
    if (++arrived == num_requests) {
      everyone_arrived = true;
      cv.notify_all();
      return;
    }
    // Assert on what the waiter observed, not on the count: a late arrival
    // would set the flag either way.
    bool released = cv.wait_for(lock, std::chrono::seconds(10),
                                [&] { return everyone_arrived; });
    EXPECT_TRUE(released) << "the batch did not run concurrently";
  };

  std::vector<SymbolLocator::Request> requests = MakeRequests(num_requests);
  std::vector<llvm::Expected<SymbolLocator::Result>> results =
      SymbolLocator::Locate(requests, FileSpecList(), /*parallel=*/true);
  for (llvm::Expected<SymbolLocator::Result> &result : results)
    if (!result)
      llvm::consumeError(result.takeError());

  EXPECT_EQ(num_requests, arrived);
}

TEST_F(SymbolLocatorTest, TheSerialBatchGivesTheSameAnswers) {
  g_object_file = m_binary;
  std::vector<SymbolLocator::Request> requests = MakeRequests(4);

  std::vector<llvm::Expected<SymbolLocator::Result>> parallel =
      SymbolLocator::Locate(requests, FileSpecList(), /*parallel=*/true);
  std::vector<llvm::Expected<SymbolLocator::Result>> serial =
      SymbolLocator::Locate(requests, FileSpecList(), /*parallel=*/false);

  ASSERT_EQ(parallel.size(), serial.size());
  for (auto [p, s] : llvm::zip_equal(parallel, serial)) {
    EXPECT_THAT_EXPECTED(p, llvm::Succeeded());
    EXPECT_THAT_EXPECTED(s, llvm::Succeeded());
    if (p && s)
      EXPECT_EQ(p->module_spec.GetFileSpec(), s->module_spec.GetFileSpec());
  }
}

TEST_F(SymbolLocatorTest, AnEmptyBatchIsNoWork) {
  std::vector<llvm::Expected<SymbolLocator::Result>> results =
      SymbolLocator::Locate({}, FileSpecList(), /*parallel=*/true);

  EXPECT_TRUE(results.empty());
  EXPECT_FALSE(g_calls.located_object_file);
}
