//===-- ClangParserTest.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "clang/Config/config.h"
#include "clang/Driver/Driver.h"

#include "Plugins/ExpressionParser/Clang/ClangHost.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-defines.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
struct ClangHostTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo> subsystems;
};
} // namespace

static std::string ComputeClangResourceDir(std::string lldb_shlib_path,
                                           bool verify = false) {
  FileSpec clang_dir;
  FileSpec lldb_shlib_spec(lldb_shlib_path);
  ComputeClangResourceDirectory(lldb_shlib_spec, clang_dir, verify);
  return clang_dir.GetPath();
}

TEST_F(ClangHostTest, ComputeClangResourceDirectory) {
#if !defined(_WIN32)
  std::string path_to_liblldb = "/foo/bar/lib/";
#else
  std::string path_to_liblldb = "C:\\foo\\bar\\lib\\";
#endif
  std::string path_to_clang_dir = clang::driver::Driver::GetResourcesPath(
      path_to_liblldb + "liblldb", CLANG_RESOURCE_DIR);
  EXPECT_EQ(ComputeClangResourceDir(path_to_liblldb), path_to_clang_dir);

  // The path doesn't really exist, so setting verify to true should make
  // ComputeClangResourceDir not give you path_to_clang_dir.
  EXPECT_NE(ComputeClangResourceDir(path_to_liblldb, true), path_to_clang_dir);
}

#if defined(__APPLE__)
TEST_F(ClangHostTest, MacOSX) {
  // This returns whatever the POSIX fallback returns.
  std::string posix = "/usr/lib/liblldb.dylib";
  EXPECT_FALSE(ComputeClangResourceDir(posix).empty());

  std::string build =
      "/lldb-macosx-x86_64/Library/Frameworks/LLDB.framework/Versions/A";
  std::string build_clang =
      "/lldb-macosx-x86_64/Library/Frameworks/LLDB.framework/Resources/Clang";
  EXPECT_EQ(ComputeClangResourceDir(build), build_clang);

  std::string xcode = "/Applications/Xcode.app/Contents/SharedFrameworks/"
                      "LLDB.framework/Versions/A";
  std::string xcode_clang =
      "/Applications/Xcode.app/Contents/Developer/Toolchains/"
      "XcodeDefault.xctoolchain/usr/lib/swift/clang";
  EXPECT_EQ(ComputeClangResourceDir(xcode), xcode_clang);

  std::string toolchain =
      "/Applications/Xcode.app/Contents/Developer/Toolchains/"
      "Swift-4.1-development-snapshot.xctoolchain/System/Library/"
      "PrivateFrameworks/LLDB.framework";
  std::string toolchain_clang =
      "/Applications/Xcode.app/Contents/Developer/Toolchains/"
      "Swift-4.1-development-snapshot.xctoolchain/usr/lib/swift/clang";
  EXPECT_EQ(ComputeClangResourceDir(toolchain), toolchain_clang);

  std::string cltools = "/Library/Developer/CommandLineTools/Library/"
                        "PrivateFrameworks/LLDB.framework";
  std::string cltools_clang =
      "/Library/Developer/CommandLineTools/Library/PrivateFrameworks/"
      "LLDB.framework/Resources/Clang";
  EXPECT_EQ(ComputeClangResourceDir(cltools), cltools_clang);

  // Test that a bogus path is detected.
  EXPECT_NE(ComputeClangResourceDir(GetInputFilePath(xcode), true),
            ComputeClangResourceDir(GetInputFilePath(xcode)));
}
#endif // __APPLE__
