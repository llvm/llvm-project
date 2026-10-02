//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Diagnostics.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteMacOSX.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/ExecutionContext.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <mutex>

using namespace lldb;
using namespace lldb_private;

namespace {
class DiagnosticsTest : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize();
    HostInfo::Initialize();
    PlatformMacOSX::Initialize();
    std::call_once(TestUtilities::g_debugger_initialize_flag,
                   []() { Debugger::Initialize(nullptr); });
    ArchSpec arch("x86_64-apple-macosx-");
    Platform::SetHostPlatform(
        PlatformRemoteMacOSX::CreateInstance(true, &arch));
  }
  void TearDown() override {
    PlatformMacOSX::Terminate();
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};

// Read a file from the bundle directory, or "" if it cannot be read.
std::string ReadBundleFile(const FileSpec &dir, llvm::StringRef name) {
  FileSpec path = dir.CopyByAppendingPathComponent(name);
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path.GetPath());
  if (!buffer)
    return "";
  return (*buffer)->getBuffer().str();
}
} // namespace

TEST_F(DiagnosticsTest, ArtifactProviderIDsAreUnique) {
  Diagnostics diagnostics;
  Diagnostics::ArtifactProviderID id0 =
      diagnostics.AddArtifactProvider("a.txt", [] { return "a"; });
  Diagnostics::ArtifactProviderID id1 =
      diagnostics.AddArtifactProvider("b.txt", [] { return "b"; });
  EXPECT_NE(id0, id1);
}

TEST_F(DiagnosticsTest, ArtifactProviderContributesToBundle) {
  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);

  Diagnostics diagnostics;
  int call_count = 0;
  Diagnostics::ArtifactProviderID id = diagnostics.AddArtifactProvider(
      "my-artifact.txt", [&call_count]() -> std::string {
        ++call_count;
        return "hello diagnostics";
      });

  ExecutionContext exe_ctx;

  // A registered provider is invoked and its content lands in the bundle and
  // in the report's attachments.
  {
    llvm::Expected<FileSpec> dir = Diagnostics::CreateUniqueDirectory();
    ASSERT_THAT_EXPECTED(dir, llvm::Succeeded());
    llvm::Expected<Diagnostics::Report> report =
        diagnostics.Collect(*debugger_sp, exe_ctx, *dir);
    ASSERT_THAT_EXPECTED(report, llvm::Succeeded());

    EXPECT_EQ(call_count, 1);
    EXPECT_TRUE(
        llvm::is_contained(report->attachments.files, "my-artifact.txt"));
    EXPECT_EQ(ReadBundleFile(*dir, "my-artifact.txt"), "hello diagnostics");

    llvm::sys::fs::remove_directories(dir->GetPath());
  }

  // After removal the provider is neither invoked nor present in the bundle.
  diagnostics.RemoveArtifactProvider(id);
  call_count = 0;
  {
    llvm::Expected<FileSpec> dir = Diagnostics::CreateUniqueDirectory();
    ASSERT_THAT_EXPECTED(dir, llvm::Succeeded());
    llvm::Expected<Diagnostics::Report> report =
        diagnostics.Collect(*debugger_sp, exe_ctx, *dir);
    ASSERT_THAT_EXPECTED(report, llvm::Succeeded());

    EXPECT_EQ(call_count, 0);
    EXPECT_FALSE(
        llvm::is_contained(report->attachments.files, "my-artifact.txt"));

    llvm::sys::fs::remove_directories(dir->GetPath());
  }

  Debugger::Destroy(debugger_sp);
}

TEST_F(DiagnosticsTest, RemoveArtifactProviderIgnoresUnknownID) {
  Diagnostics diagnostics;
  Diagnostics::ArtifactProviderID id =
      diagnostics.AddArtifactProvider("kept.txt", [] { return "kept"; });

  // Removing an id that was never handed out must not disturb other providers.
  diagnostics.RemoveArtifactProvider(id + 1);

  DebuggerSP debugger_sp = Debugger::CreateInstance();
  ASSERT_TRUE(debugger_sp);
  ExecutionContext exe_ctx;
  llvm::Expected<FileSpec> dir = Diagnostics::CreateUniqueDirectory();
  ASSERT_THAT_EXPECTED(dir, llvm::Succeeded());
  llvm::Expected<Diagnostics::Report> report =
      diagnostics.Collect(*debugger_sp, exe_ctx, *dir);
  ASSERT_THAT_EXPECTED(report, llvm::Succeeded());

  EXPECT_TRUE(llvm::is_contained(report->attachments.files, "kept.txt"));

  llvm::sys::fs::remove_directories(dir->GetPath());
  Debugger::Destroy(debugger_sp);
}
