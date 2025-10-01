//===-- SourceManagerTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/SourceManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Utility/SupportFile.h"
#include "gtest/gtest.h"

#include "TestingSupport/MockTildeExpressionResolver.h"

using namespace lldb;
using namespace lldb_private;

class SourceFileCache : public ::testing::Test {
public:
  void SetUp() override {
    FileSystem::Initialize(std::unique_ptr<TildeExpressionResolver>(
        new MockTildeExpressionResolver("Jonas", "/jonas")));
  }
  void TearDown() override { FileSystem::Terminate(); }
};

TEST_F(SourceFileCache, FindSourceFileFound) {
  SourceManager::SourceFileCache cache;

  // Insert: foo
  FileSpec foo_file_spec("foo");
  auto foo_file_sp = std::make_shared<SourceManager::File>(
      std::make_shared<SupportFile>(foo_file_spec), lldb::DebuggerSP());
  cache.AddSourceFile(foo_file_spec, foo_file_sp);

  // Query: foo, expect found.
  FileSpec another_foo_file_spec("foo");
  ASSERT_EQ(cache.FindSourceFile(another_foo_file_spec), foo_file_sp);
}

TEST_F(SourceFileCache, FindSourceFileNotFound) {
  SourceManager::SourceFileCache cache;

  // Insert: foo
  FileSpec foo_file_spec("foo");
  auto foo_file_sp = std::make_shared<SourceManager::File>(
      std::make_shared<SupportFile>(foo_file_spec), lldb::DebuggerSP());
  cache.AddSourceFile(foo_file_spec, foo_file_sp);

  // Query: bar, expect not found.
  FileSpec bar_file_spec("bar");
  ASSERT_EQ(cache.FindSourceFile(bar_file_spec), nullptr);
}

TEST_F(SourceFileCache, FindSourceFileByUnresolvedPath) {
  SourceManager::SourceFileCache cache;

  FileSpec foo_file_spec("~/foo");

  // Mimic the resolution in SourceManager::GetFile.
  FileSpec resolved_foo_file_spec = foo_file_spec;
  FileSystem::Instance().Resolve(resolved_foo_file_spec);

  // Create the file with the resolved file spec.
  auto foo_file_sp = std::make_shared<SourceManager::File>(
      std::make_shared<SupportFile>(resolved_foo_file_spec),
      lldb::DebuggerSP());

  // Cache the result with the unresolved file spec.
  cache.AddSourceFile(foo_file_spec, foo_file_sp);

  // Query the unresolved path.
  EXPECT_EQ(cache.FindSourceFile(FileSpec("~/foo")), foo_file_sp);

  // Query the resolved path.
  EXPECT_EQ(cache.FindSourceFile(FileSpec("/jonas/foo")), foo_file_sp);
}
