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
using namespace llvm::sys;

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
  /// Unique temporary directory in which all created filesystem entities must
  /// be placed. It is removed at the end of the test suite.
  llvm::SmallString<128> m_base_dir;

  void SetUp() override {
    // Get the name of the current test. To prevent that by chance two tests
    // get the same temporary directory if createUniqueDirectory fails.
    auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    ASSERT_TRUE(test_info != nullptr);
    std::string name = test_info->name();
    ASSERT_NO_ERROR(
        fs::createUniqueDirectory("SwiftASTCtx-" + name, m_base_dir));
  }

  void TearDown() override {
    ASSERT_NO_ERROR(fs::remove_directories(m_base_dir));
  }

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

  static std::string GetResourceDir(llvm::StringRef platform_sdk_path,
                                    std::string swift_dir,
                                    std::string swift_stdlib_os_dir,
                                    std::string xcode_contents_path,
                                    std::string toolchain_path,
                                    std::string cl_tools_path) {
    return SwiftASTContext::GetResourceDir(
        platform_sdk_path, swift_dir, swift_stdlib_os_dir, xcode_contents_path,
        toolchain_path, cl_tools_path);
  }
  static std::string GetSwiftStdlibOSDir(const llvm::Triple &target,
                                         const llvm::Triple &host) {
    return SwiftASTContext::GetSwiftStdlibOSDir(target, host);
  }
};

TEST_F(TestSwiftASTContext, ResourceDir) {
  const char *paths[] = {
      // Toolchains.
      "/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/"
      "lib/swift/macosx",

      "/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/"
      "lib/swift/iphoneos",

      "/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/"
      "lib/swift/iphonesimulator",

      // SDKs.
      "/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/"
      "MacOSX10.13.sdk/usr",

      "/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/"
      "SDKs/"
      "iPhoneOS11.3.sdk/usr",

      "/Xcode.app/Contents/Developer/Platforms/iPhoneSimulator.platform/"
      "Developer/SDKs/iPhoneSimulatorOS12.0.sdk/usr",

      "/Xcode.app/Contents/Developer/Platforms/Linux.platform/Developer/SDKs/"
      "Linux.sdk/usr/lib/swift/linux",

      // Custom toolchains.
      "/Xcode.app/Contents/Developer/Toolchains/"
      "Swift-4.1-development-snapshot.xctoolchain/usr/lib/swift/macosx",

      // CLTools.
      "/Library/Developer/CommandLineTools/usr/lib/swift/macosx",

      // Local builds.
      "/build/LLDB.framework/Resources/Swift/clang",
  };

  using SmallString = llvm::SmallString<256>;
  std::vector<std::string> abs_paths;
  for (auto dir : paths) {
    SmallString path = m_base_dir.str();
    path::append(path, dir);
    ASSERT_NO_ERROR(fs::create_directories(path));
    abs_paths.push_back(std::string(path));
  }

  llvm::StringRef macosx_sdk = path::parent_path(abs_paths[3]);
  llvm::StringRef ios_sdk = path::parent_path(abs_paths[4]);
  llvm::StringRef iossim_sdk = path::parent_path(abs_paths[5]);
  llvm::StringRef cross_sdk = path::parent_path(
      path::parent_path(path::parent_path(path::parent_path(abs_paths[6]))));

  SmallString swift_dir = m_base_dir.str();
  path::append(swift_dir,
      "/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Resources/Swift");
  SmallString xcode_contents = m_base_dir.str();
  path::append(xcode_contents, "/Xcode.app/Contents");
  SmallString toolchain = m_base_dir.str();
  path::append(toolchain, "/Xcode.app/Contents/Developer/Toolchains");
  SmallString cl_tools = m_base_dir.str();
  path::append(cl_tools, "/Library/Developer/CommandLineTools");

  SmallString tc_rdir = m_base_dir.str();
  llvm::sys::path::append(tc_rdir, "/Xcode.app/Contents/Developer/Toolchains/"
                                   "XcodeDefault.xctoolchain/usr/lib/swift");

  auto GetResourceDir = [&](const char *triple_string,
                            llvm::StringRef sdk_path) {
    llvm::Triple host("x86_64-apple-macosx10.14");
    llvm::Triple target(triple_string);
    return SwiftASTContextTester::GetResourceDir(
        sdk_path,
        SwiftASTContextTester::GetSwiftStdlibOSDir(target, host),
        std::string(swift_dir), std::string(xcode_contents),
        std::string(toolchain), std::string(cl_tools));
  };

  EXPECT_EQ(GetResourceDir("x86_64-apple-macosx10.14", macosx_sdk),
            tc_rdir.str());
  EXPECT_EQ(GetResourceDir("x86_64-apple-darwin", macosx_sdk), tc_rdir);
  EXPECT_EQ(GetResourceDir("aarch64-apple-ios11.3", ios_sdk), tc_rdir);
  // Old-style simulator triple with missing environment.
  EXPECT_EQ(GetResourceDir("x86_64-apple-ios11.3", iossim_sdk), tc_rdir);
  EXPECT_EQ(GetResourceDir("x86_64-apple-ios11.3-simulator", iossim_sdk),
            tc_rdir);
  EXPECT_EQ(GetResourceDir("x86_64-unknown-linux", cross_sdk),
            path::parent_path(abs_paths[6]));

  // Version is too low, but we still expect a valid resource directory.
  EXPECT_EQ(GetResourceDir("x86_64-apple-ios11.0", ios_sdk), tc_rdir);
  std::string s = GetResourceDir("armv7k-apple-watchos4.0", ios_sdk);
  EXPECT_NE(GetResourceDir("armv7k-apple-watchos4.0", ios_sdk), "");

  // Custom toolchain.
  toolchain = path::parent_path(
      path::parent_path(path::parent_path(path::parent_path(abs_paths[7]))));
  std::string custom_tc = path::parent_path(abs_paths[7]).str();
  EXPECT_EQ(GetResourceDir("x86_64-apple-macosx", macosx_sdk), custom_tc);

  // CLTools.
  xcode_contents = "";
  toolchain = "";
  std::string cl_tools_rd = path::parent_path(abs_paths[8]).str();
  EXPECT_EQ(GetResourceDir("x86_64-apple-macosx", macosx_sdk), cl_tools_rd);

  // Local builds.
  swift_dir = path::parent_path(abs_paths[9]);
  EXPECT_EQ(GetResourceDir("x86_64-apple-macosx", macosx_sdk), swift_dir);
}

TEST_F(TestSwiftASTContext, IsNonTriviallyManagedReferenceType) {
#ifndef NDEBUG
  // The mock constructor is only available in asserts mode.
  SwiftASTContext::NonTriviallyManagedReferenceStrategy strategy;
  SwiftASTContext context;
  CompilerType t(&context, nullptr);
  EXPECT_FALSE(SwiftASTContext::IsNonTriviallyManagedReferenceType(t, strategy,
                                                                   nullptr));
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

namespace {
  const std::vector<std::string> duplicated_flags = {
    "-DMACRO1", "-D", "MACRO1", "-UMACRO2", "-U", "MACRO2",
    "-I/path1", "-I", "/path1", "-F/path2", "-F", "/path2",
    "-fmodule-map-file=/path3", "-fmodule-map-file=/path3",
    "-F/path2", "-F", "/path2", "-I/path1", "-I", "/path1",
    "-UMACRO2", "-U", "MACRO2", "-DMACRO1", "-D", "MACRO1",
  };
  const std::vector<std::string> uniqued_flags = {
    "-DMACRO1", "-UMACRO2", "-I/path1", "-F/path2", "-fmodule-map-file=/path3"
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
