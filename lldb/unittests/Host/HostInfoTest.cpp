//===-- HostInfoTest.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/HostInfo.h"
#include "lldb/lldb-defines.h"
#include "TestingSupport/TestUtilities.h"
#include "gtest/gtest.h"

#ifdef __APPLE__
#include "lldb/Host/macosx/HostInfoMacOSX.h"
#endif

using namespace lldb_private;
using namespace llvm;

namespace {
class HostInfoTest: public ::testing::Test {
  public:
    void SetUp() override { HostInfo::Initialize(); }
    void TearDown() override { HostInfo::Terminate(); }
};
}

TEST_F(HostInfoTest, GetAugmentedArchSpec) {
  // Fully specified triple should not be changed.
  ArchSpec spec = HostInfo::GetAugmentedArchSpec("x86_64-pc-linux-gnu");
  EXPECT_EQ(spec.GetTriple().getTriple(), "x86_64-pc-linux-gnu");

  // Same goes if we specify at least one of (os, vendor, env).
  spec = HostInfo::GetAugmentedArchSpec("x86_64-pc");
  EXPECT_EQ(spec.GetTriple().getTriple(), "x86_64-pc");

  // But if we specify only an arch, we should fill in the rest from the host.
  spec = HostInfo::GetAugmentedArchSpec("x86_64");
  Triple triple(sys::getDefaultTargetTriple());
  EXPECT_EQ(spec.GetTriple().getArch(), Triple::x86_64);
  EXPECT_EQ(spec.GetTriple().getOS(), triple.getOS());
  EXPECT_EQ(spec.GetTriple().getVendor(), triple.getVendor());
  EXPECT_EQ(spec.GetTriple().getEnvironment(), triple.getEnvironment());

  // Test LLDB_ARCH_DEFAULT
  EXPECT_EQ(HostInfo::GetAugmentedArchSpec(LLDB_ARCH_DEFAULT).GetTriple(),
            HostInfo::GetArchitecture(HostInfo::eArchKindDefault).GetTriple());
}


#ifdef __APPLE__

struct HostInfoMacOSXTest : public HostInfoMacOSX {
  static std::string ComputeClangDir(std::string lldb_shlib_path,
                                     bool verify = false) {
    FileSpec clang_dir;
    FileSpec lldb_shlib_spec(lldb_shlib_path, false);
    ComputeClangDirectory(lldb_shlib_spec, clang_dir, verify);
    return clang_dir.GetPath();
  }
};


TEST_F(HostInfoTest, MacOSX) {
  // This returns whatever the POSIX fallback returns.
  std::string posix = "/usr/lib/liblldb.dylib";
  EXPECT_FALSE(HostInfoMacOSXTest::ComputeClangDir(posix).empty());

  std::string build =
    "/lldb-macosx-x86_64/Library/Frameworks/LLDB.framework/Versions/A";
  std::string build_clang =
    "/lldb-macosx-x86_64/Library/Frameworks/LLDB.framework/Resources/Clang";
  EXPECT_EQ(HostInfoMacOSXTest::ComputeClangDir(build), build_clang);

  std::string xcode =
    "/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Versions/A";
  std::string xcode_clang =
    "/Applications/Xcode.app/Contents/Developer/Toolchains/"
    "XcodeDefault.xctoolchain/usr/lib/swift/clang";
  EXPECT_EQ(HostInfoMacOSXTest::ComputeClangDir(xcode), xcode_clang);

  std::string toolchain =
      "/Applications/Xcode.app/Contents/Developer/Toolchains/"
      "Swift-4.1-development-snapshot.xctoolchain/System/Library/"
      "PrivateFrameworks/LLDB.framework";
  std::string toolchain_clang =
      "/Applications/Xcode.app/Contents/Developer/Toolchains/"
      "Swift-4.1-development-snapshot.xctoolchain/usr/lib/swift/clang";
  EXPECT_EQ(HostInfoMacOSXTest::ComputeClangDir(toolchain), toolchain_clang);

  // Test that a bogus path is detected.
  EXPECT_NE(HostInfoMacOSXTest::ComputeClangDir(GetInputFilePath(xcode), true),
            HostInfoMacOSXTest::ComputeClangDir(GetInputFilePath(xcode)));
}
#endif
