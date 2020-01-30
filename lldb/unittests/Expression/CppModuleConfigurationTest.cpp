//===-- CppModuleConfigurationTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ExpressionParser/Clang/CppModuleConfiguration.h"
#include "Plugins/ExpressionParser/Clang/ClangHost.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
struct CppModuleConfigurationTest : public testing::Test {
  SubsystemRAII<FileSystem, HostInfo> subsystems;
};
} // namespace

/// Returns the Clang resource include directory.
static std::string ResourceInc() {
  llvm::SmallString<256> resource_dir;
  llvm::sys::path::append(resource_dir, GetClangResourceDir().GetPath(),
                          "include");
  return std::string(resource_dir);
}

/// Utility function turningn a list of paths into a FileSpecList.
static FileSpecList makeFiles(llvm::ArrayRef<std::string> paths) {
  FileSpecList result;
  for (const std::string &path : paths)
    result.Append(FileSpec(path, FileSpec::Style::posix));
  return result;
}

TEST_F(CppModuleConfigurationTest, Linux) {
  // Test the average Linux configuration.
  std::string libcpp = "/usr/include/c++/v1";
  std::string usr = "/usr/include";
  CppModuleConfiguration config(
      makeFiles({usr + "/bits/types.h", libcpp + "/vector"}));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre(libcpp, ResourceInc(), usr));
}

TEST_F(CppModuleConfigurationTest, Sysroot) {
  // Test that having a sysroot for the whole system works fine.
  std::string libcpp = "/home/user/sysroot/usr/include/c++/v1";
  std::string usr = "/home/user/sysroot/usr/include";
  CppModuleConfiguration config(
      makeFiles({usr + "/bits/types.h", libcpp + "/vector"}));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre(libcpp, ResourceInc(), usr));
}

TEST_F(CppModuleConfigurationTest, LinuxLocalLibCpp) {
  // Test that a locally build libc++ is detected.
  std::string libcpp = "/home/user/llvm-build/include/c++/v1";
  std::string usr = "/usr/include";
  CppModuleConfiguration config(
      makeFiles({usr + "/bits/types.h", libcpp + "/vector"}));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre(libcpp, ResourceInc(), usr));
}

TEST_F(CppModuleConfigurationTest, UnrelatedLibrary) {
  // Test that having an unrelated library in /usr/include doesn't break.
  std::string libcpp = "/home/user/llvm-build/include/c++/v1";
  std::string usr = "/usr/include";
  CppModuleConfiguration config(makeFiles(
      {usr + "/bits/types.h", libcpp + "/vector", usr + "/boost/vector"}));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre(libcpp, ResourceInc(), usr));
}

TEST_F(CppModuleConfigurationTest, Xcode) {
  // Test detection of libc++ coming from Xcode with generic platform names.
  std::string p = "/Applications/Xcode.app/Contents/Developer/";
  std::string libcpp = p + "Toolchains/B.xctoolchain/usr/include/c++/v1";
  std::string usr =
      p + "Platforms/A.platform/Developer/SDKs/OSVers.sdk/usr/include";
  CppModuleConfiguration config(
      makeFiles({libcpp + "/unordered_map", usr + "/stdio.h"}));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre(libcpp, ResourceInc(), usr));
}

TEST_F(CppModuleConfigurationTest, LibCppV2) {
  // Test that a "v2" of libc++ is still correctly detected.
  CppModuleConfiguration config(
      makeFiles({"/usr/include/bits/types.h", "/usr/include/c++/v2/vector"}));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre("/usr/include/c++/v2", ResourceInc(),
                                   "/usr/include"));
}

TEST_F(CppModuleConfigurationTest, UnknownLibCppFile) {
  // Test that having some unknown file in the libc++ path doesn't break
  // anything.
  CppModuleConfiguration config(makeFiles(
      {"/usr/include/bits/types.h", "/usr/include/c++/v1/non_existing_file"}));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre("/usr/include/c++/v1", ResourceInc(),
                                   "/usr/include"));
}

TEST_F(CppModuleConfigurationTest, MissingUsrInclude) {
  // Test that we don't load 'std' if we can't find the C standard library.
  CppModuleConfiguration config(makeFiles({"/usr/include/c++/v1/vector"}));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre());
  EXPECT_THAT(config.GetIncludeDirs(), testing::ElementsAre());
}

TEST_F(CppModuleConfigurationTest, MissingLibCpp) {
  // Test that we don't load 'std' if we don't have a libc++.
  CppModuleConfiguration config(makeFiles({"/usr/include/bits/types.h"}));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre());
  EXPECT_THAT(config.GetIncludeDirs(), testing::ElementsAre());
}

TEST_F(CppModuleConfigurationTest, IgnoreLibStdCpp) {
  // Test that we don't do anything bad when we encounter libstdc++ paths.
  CppModuleConfiguration config(makeFiles(
      {"/usr/include/bits/types.h", "/usr/include/c++/8.0.1/vector"}));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre());
  EXPECT_THAT(config.GetIncludeDirs(), testing::ElementsAre());
}

TEST_F(CppModuleConfigurationTest, AmbiguousCLib) {
  // Test that we don't do anything when we are not sure where the
  // right C standard library is.
  CppModuleConfiguration config(
      makeFiles({"/usr/include/bits/types.h", "/usr/include/c++/v1/vector",
                 "/sysroot/usr/include/bits/types.h"}));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre());
  EXPECT_THAT(config.GetIncludeDirs(), testing::ElementsAre());
}

TEST_F(CppModuleConfigurationTest, AmbiguousLibCpp) {
  // Test that we don't do anything when we are not sure where the
  // right libc++ is.
  CppModuleConfiguration config(
      makeFiles({"/usr/include/bits/types.h", "/usr/include/c++/v1/vector",
                 "/usr/include/c++/v2/vector"}));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre());
  EXPECT_THAT(config.GetIncludeDirs(), testing::ElementsAre());
}
