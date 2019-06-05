//===-- SwiftParserTest.cpp --------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Plugins/ExpressionParser/Swift/SwiftHost.h"
#include "TestingSupport/TestUtilities.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/lldb-defines.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
struct SwiftHostTest : public testing::Test {
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
  ComputeSwiftResourceDirectory(lldb_shlib_spec, swift_dir, verify);
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
#endif
