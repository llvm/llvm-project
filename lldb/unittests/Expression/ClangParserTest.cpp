//===-- ClangParserTest.cpp --------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Plugins/ExpressionParser/Clang/ClangHost.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-defines.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
struct ClangHostTest : public testing::Test {
  static void SetUpTestCase() { HostInfo::Initialize(); }
  static void TearDownTestCase() { HostInfo::Terminate(); }
};
} // namespace

#ifdef __APPLE__
static std::string ComputeClangDir(std::string lldb_shlib_path,
                                   bool verify = false) {
  FileSpec clang_dir;
  FileSpec lldb_shlib_spec(lldb_shlib_path, false);
  ComputeClangDirectory(lldb_shlib_spec, clang_dir, verify);
  return clang_dir.GetPath();
}

TEST_F(ClangHostTest, MacOSX) {
  // This returns whatever the POSIX fallback returns.
  std::string posix = "/usr/lib/liblldb.dylib";
  EXPECT_FALSE(ComputeClangDir(posix).empty());

  std::string build =
      "/lldb-macosx-x86_64/Library/Frameworks/LLDB.framework/Versions/A";
  std::string build_clang =
      "/lldb-macosx-x86_64/Library/Frameworks/LLDB.framework/Resources/Clang";
  EXPECT_EQ(ComputeClangDir(build), build_clang);

  std::string xcode = "/Applications/Xcode.app/Contents/SharedFrameworks/"
                      "LLDB.framework/Versions/A";
  std::string xcode_clang =
      "/Applications/Xcode.app/Contents/Developer/Toolchains/"
      "XcodeDefault.xctoolchain/usr/lib/swift/clang";
  EXPECT_EQ(ComputeClangDir(xcode), xcode_clang);

  std::string toolchain =
      "/Applications/Xcode.app/Contents/Developer/Toolchains/"
      "Swift-4.1-development-snapshot.xctoolchain/System/Library/"
      "PrivateFrameworks/LLDB.framework";
  std::string toolchain_clang =
      "/Applications/Xcode.app/Contents/Developer/Toolchains/"
      "Swift-4.1-development-snapshot.xctoolchain/usr/lib/swift/clang";
  EXPECT_EQ(ComputeClangDir(toolchain), toolchain_clang);

  std::string cltools = "/Library/Developer/CommandLineTools/Library/"
                        "PrivateFrameworks/LLDB.framework";
  std::string cltools_clang =
      "/Library/Developer/CommandLineTools/Library/PrivateFrameworks/"
      "LLDB.framework/Resources/Clang";
  EXPECT_EQ(ComputeClangDir(cltools), cltools_clang);

  // Test that a bogus path is detected.
  EXPECT_NE(ComputeClangDir(GetInputFilePath(xcode), true),
            ComputeClangDir(GetInputFilePath(xcode)));
}
#endif
