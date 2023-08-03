//===-- HostInfoSwiftTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/TestUtilities.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-defines.h"
#include "llvm/Support/FileSystem.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace llvm::sys;

namespace {
struct SwiftHostTest : public testing::Test {
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
        fs::createUniqueDirectory("SwiftHostTest-" + name, m_base_dir));
  }

  void TearDown() override {
    ASSERT_NO_ERROR(fs::remove_directories(m_base_dir));
  }

  static void SetUpTestCase() {
    FileSystem::Initialize();
    HostInfo::Initialize();
  }
  static void TearDownTestCase() {
    HostInfo::Terminate();
    FileSystem::Terminate();
  }
};
} // namespace

static std::string
ComputeSwiftResourceDirectoryHelper(std::string lldb_shlib_path,
                                    bool verify = false) {
  FileSpec swift_dir;
  FileSpec lldb_shlib_spec(lldb_shlib_path);
  HostInfo::ComputeSwiftResourceDirectory(lldb_shlib_spec, swift_dir, verify);
  return swift_dir.GetPath();
}

TEST_F(SwiftHostTest, ComputeSwiftResourceDirectory) {
#if !defined(_WIN32)
  std::string path_to_liblldb = "/foo/bar/lib";
  std::string path_to_swift_dir = "/foo/bar/lib/swift";
#else
  std::string path_to_liblldb = "C:\\foo\\bar\\bin";
  std::string path_to_swift_dir = "C:\\foo\\bar\\lib\\swift";
#endif
  EXPECT_EQ(ComputeSwiftResourceDirectoryHelper(path_to_liblldb),
            path_to_swift_dir);
  EXPECT_NE(ComputeSwiftResourceDirectoryHelper(path_to_liblldb),
            ComputeSwiftResourceDirectoryHelper(path_to_liblldb, true));
  EXPECT_TRUE(ComputeSwiftResourceDirectoryHelper("").empty());
}

#if defined(__APPLE__)
TEST_F(SwiftHostTest, MacOSX) {
  // test for LLDB.framework
  std::string path_to_liblldb = "/foo/bar/lib/LLDB.framework";
  std::string path_to_swift_dir = "/foo/bar/lib/LLDB.framework/Resources/Swift";
  EXPECT_EQ(ComputeSwiftResourceDirectoryHelper(path_to_liblldb),
            path_to_swift_dir);
  path_to_liblldb = "/foo/bar/lib/LLDB.framework/foo/bar";
  path_to_swift_dir = "/foo/bar/lib/LLDB.framework/Resources/Swift";
  EXPECT_EQ(ComputeSwiftResourceDirectoryHelper(path_to_liblldb),
            path_to_swift_dir);
}

TEST_F(SwiftHostTest, ResourceDir) {
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
    return HostInfoMacOSX::DetectSwiftResourceDir(
        sdk_path, HostInfoMacOSX::GetSwiftStdlibOSDir(target, host),
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
#endif
