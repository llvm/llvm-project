//===-- TestSwiftASTContext.cpp -------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"

using namespace lldb;
using namespace lldb_private;

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    llvm::SmallString<128> MessageStorage;                                     \
    llvm::raw_svector_ostream Message(MessageStorage);                         \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }

struct TestSwiftASTContext : public testing::Test {
  static void SetUpTestCase() {
    FileSystem::Initialize();
    HostInfo::Initialize();  }

  static void TearDownTestCase() {
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};

struct SwiftASTContextTester : public SwiftASTContext {
  #ifndef NDEBUG
    SwiftASTContextTester() : SwiftASTContext() {}
  #endif

  TypeSystemSwiftTypeRef &GetTypeSystemSwiftTypeRef() override {
    return m_typeref_typesystem;
  }

  TypeSystemSwiftTypeRef m_typeref_typesystem;
};

TEST_F(TestSwiftASTContext, IsNonTriviallyManagedReferenceType) {
#ifndef NDEBUG
  // The mock constructor is only available in asserts mode.
  auto context = std::make_shared<SwiftASTContextTester>();
  EXPECT_FALSE(context->GetNonTriviallyManagedReferenceKind(nullptr));
#endif
}

TEST_F(TestSwiftASTContext, SwiftFriendlyTriple) {
  EXPECT_EQ(SwiftASTContext::GetSwiftFriendlyTriple(
                llvm::Triple("x86_64-apple-macosx")),
            llvm::Triple("x86_64-apple-macosx"));
  EXPECT_EQ(SwiftASTContext::GetSwiftFriendlyTriple(
                llvm::Triple("x86_64h-apple-macosx")),
            llvm::Triple("x86_64-apple-macosx"));
  EXPECT_EQ(SwiftASTContext::GetSwiftFriendlyTriple(
                llvm::Triple("aarch64-apple-macosx")),
            llvm::Triple("arm64-apple-macosx"));
  EXPECT_EQ(SwiftASTContext::GetSwiftFriendlyTriple(
                llvm::Triple("aarch64_32-apple-watchos")),
            llvm::Triple("arm64_32-apple-watchos"));
  EXPECT_EQ(SwiftASTContext::GetSwiftFriendlyTriple(
                llvm::Triple("aarch64-unknown-linux")),
            llvm::Triple("aarch64-unknown-linux-gnu"));
}

TEST_F(TestSwiftASTContext, ApplyWorkingDir) {
  std::string abs_working_dir = "/abs/dir";
  std::string rel_working_dir = "rel/dir";

  // non-include option should not apply working dir
  llvm::SmallString<128> non_include_flag("-non-include-flag");
  SwiftASTContext::ApplyWorkingDir(non_include_flag, abs_working_dir);
  EXPECT_EQ(non_include_flag, llvm::SmallString<128>("-non-include-flag"));

  // absolute paths should not apply working dir
  llvm::SmallString<128> abs_path("/abs/path");
  SwiftASTContext::ApplyWorkingDir(abs_path, abs_working_dir);
  EXPECT_EQ(abs_path, llvm::SmallString<128>("/abs/path"));

  llvm::SmallString<128> single_arg_abs_path(
      "-fmodule-map-file=/module/map/path");
  SwiftASTContext::ApplyWorkingDir(single_arg_abs_path, abs_working_dir);
  EXPECT_EQ(single_arg_abs_path,
            llvm::SmallString<128>("-fmodule-map-file=/module/map/path"));

  // relative paths apply working dir
  llvm::SmallString<128> rel_path("rel/path");
  SwiftASTContext::ApplyWorkingDir(rel_path, abs_working_dir);
  EXPECT_EQ(rel_path, llvm::SmallString<128>("/abs/dir/rel/path"));

  rel_path = llvm::SmallString<128>("rel/path");
  SwiftASTContext::ApplyWorkingDir(rel_path, rel_working_dir);
  EXPECT_EQ(rel_path, llvm::SmallString<128>("rel/dir/rel/path"));

  // single arg include option applies working dir
  llvm::SmallString<128> single_arg_rel_path(
      "-fmodule-map-file=module.modulemap");
  SwiftASTContext::ApplyWorkingDir(single_arg_rel_path, abs_working_dir);
  EXPECT_EQ(
      single_arg_rel_path,
      llvm::SmallString<128>("-fmodule-map-file=/abs/dir/module.modulemap"));

  single_arg_rel_path =
      llvm::SmallString<128>("-fmodule-map-file=module.modulemap");
  SwiftASTContext::ApplyWorkingDir(single_arg_rel_path, rel_working_dir);
  EXPECT_EQ(
      single_arg_rel_path,
      llvm::SmallString<128>("-fmodule-map-file=rel/dir/module.modulemap"));

  // fmodule-file needs to handle different cases:
  //  -fmodule-file=path/to/pcm
  //  -fmodule-file=name=path/to/pcm
  llvm::SmallString<128> module_file_abs_path(
      "-fmodule-file=/some/dir/module.pcm");
  SwiftASTContext::ApplyWorkingDir(module_file_abs_path, abs_working_dir);
  EXPECT_EQ(module_file_abs_path,
            llvm::SmallString<128>("-fmodule-file=/some/dir/module.pcm"));

  llvm::SmallString<128> module_file_rel_path(
      "-fmodule-file=relpath/module.pcm");
  SwiftASTContext::ApplyWorkingDir(module_file_rel_path, abs_working_dir);
  EXPECT_EQ(
      module_file_rel_path,
      llvm::SmallString<128>("-fmodule-file=/abs/dir/relpath/module.pcm"));

  llvm::SmallString<128> module_file_with_name_abs_path(
      "-fmodule-file=modulename=/some/dir/module.pcm");
  SwiftASTContext::ApplyWorkingDir(module_file_with_name_abs_path,
                                   abs_working_dir);
  EXPECT_EQ(
      module_file_with_name_abs_path,
      llvm::SmallString<128>("-fmodule-file=modulename=/some/dir/module.pcm"));

  llvm::SmallString<128> module_file_with_name_rel_path(
      "-fmodule-file=modulename=relpath/module.pcm");
  SwiftASTContext::ApplyWorkingDir(module_file_with_name_rel_path,
                                   abs_working_dir);
  EXPECT_EQ(module_file_with_name_rel_path,
            llvm::SmallString<128>(
                "-fmodule-file=modulename=/abs/dir/relpath/module.pcm"));
}

namespace {
const std::vector<std::string> duplicated_flags = {
    "-DMACRO1",
    "-D",
    "MACRO1",
    "-UMACRO2",
    "-U",
    "MACRO2",
    "-I/path1",
    "-I",
    "/path1",
    "-F/path2",
    "-F",
    "/path2",
    "-fmodule-map-file=/path3",
    "-fmodule-map-file=/path3",
    "-F/path2",
    "-F",
    "/path2",
    "-I/path1",
    "-I",
    "/path1",
    "-UMACRO2",
    "-U",
    "MACRO2",
    "-DMACRO1",
    "-D",
    "MACRO1",
    "-fmodule-file=/path/to/pcm",
    "-fmodule-file=/path/to/pcm",
    "-fmodule-file=modulename=/path/to/pcm",
    "-fmodule-file=modulename=/path/to/pcm",
};
const std::vector<std::string> uniqued_flags = {
    "-DMACRO1",
    "-UMACRO2",
    "-I/path1",
    "-F/path2",
    "-fmodule-map-file=/path3",
    "-fmodule-file=/path/to/pcm",
    "-fmodule-file=modulename=/path/to/pcm",
};
} // namespace

TEST(ClangArgs, UniquingCollisionWithExistingFlags) {
  const std::vector<std::string> source = duplicated_flags;
  std::vector<std::string> dest = uniqued_flags;
  SwiftASTContext::AddExtraClangArgs(source, dest);

  EXPECT_EQ(dest, uniqued_flags);
}

TEST(ClangArgs, UniquingCollisionWithAddedFlags) {
  const std::vector<std::string> source = duplicated_flags;
  std::vector<std::string> dest;
  SwiftASTContext::AddExtraClangArgs(source, dest);

  EXPECT_EQ(dest, uniqued_flags);
}

TEST(ClangArgs, DoubleDash) {
  // -v with all currently ignored arguments following.
  const std::vector<std::string> source{"-v", "--", "-Werror", ""};
  std::vector<std::string> dest;
  SwiftASTContext::AddExtraClangArgs(source, dest);

  // Check that all ignored arguments got removed.
  EXPECT_EQ(dest, std::vector<std::string>({"-v"}));
}
