//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/SymbolLocator.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"
#include "lldb/Utility/FileSpecList.h"

#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"

#include "gtest/gtest.h"

using namespace lldb;
using namespace lldb_private;

namespace {

/// Which steps of the search ran, so a test can tell where an answer came from.
struct LocatorCalls {
  bool located_symbol_file = false;
  bool located_object_file = false;
  bool downloaded = false;
};

LocatorCalls g_calls;

/// What the object file locator claims to have found, if anything.
std::optional<FileSpec> g_object_file;

/// What the symbol file locator claims to have found, if anything.
std::optional<FileSpec> g_symbol_file;

/// When set, the symbol server fails the way a failure to launch it does, with
/// an errno rather than a message.
bool g_symbol_server_errno = false;

std::optional<FileSpec> LocateExecutableSymbolFile(const ModuleSpec &,
                                                   const FileSpecList &) {
  g_calls.located_symbol_file = true;
  return g_symbol_file;
}

std::optional<ModuleSpec> LocateExecutableObjectFile(const ModuleSpec &spec) {
  g_calls.located_object_file = true;
  if (!g_object_file)
    return {};
  ModuleSpec located(spec);
  located.GetFileSpec() = *g_object_file;
  return located;
}

bool DownloadObjectAndSymbolFile(ModuleSpec &, Status &error, bool force_lookup,
                                 bool) {
  g_calls.downloaded = true;
  if (!force_lookup)
    return false;
  if (g_symbol_server_errno)
    error = Status(std::make_error_code(std::errc::too_many_files_open));
  else
    error = Status::FromErrorString("the symbol server said no");
  return false;
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
    // Locate reports a binary it cannot find as an error, so a test that wants
    // a hit has to point the fake locator at a file that exists.
    FileSystem::Initialize(m_fs);
    HostInfo::Initialize();
    m_fs->addFileNoOwn(m_binary.GetPath(), 0, m_empty_buffer);
    m_fs->addFileNoOwn(m_symbols.GetPath(), 0, m_empty_buffer);

    g_calls = LocatorCalls();
    g_object_file = std::nullopt;
    g_symbol_file = std::nullopt;
    g_symbol_server_errno = false;
    ASSERT_TRUE(PluginManager::RegisterPlugin(
        "test", "test symbol locator", CreateSymbolLocator,
        LocateExecutableObjectFile, LocateExecutableSymbolFile,
        DownloadObjectAndSymbolFile));
  }

  void TearDown() override {
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
