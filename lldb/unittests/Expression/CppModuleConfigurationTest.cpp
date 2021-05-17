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
#include "llvm/Support/SmallVectorMemoryBuffer.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace lldb_private;

namespace {
struct CppModuleConfigurationTest : public testing::Test {
  llvm::MemoryBufferRef m_empty_buffer;
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> m_fs;

  CppModuleConfigurationTest()
      : m_empty_buffer("", "<empty buffer>"),
        m_fs(new llvm::vfs::InMemoryFileSystem()) {}

  void SetUp() override {
    FileSystem::Initialize(m_fs);
    HostInfo::Initialize();
  }

  void TearDown() override {
    HostInfo::Terminate();
    FileSystem::Terminate();
  }

  /// Utility function turning a list of paths into a FileSpecList.
  FileSpecList makeFiles(llvm::ArrayRef<std::string> paths) {
    FileSpecList result;
    for (const std::string &path : paths) {
      result.Append(FileSpec(path, FileSpec::Style::posix));
      if (!m_fs->addFileNoOwn(path, static_cast<time_t>(0), m_empty_buffer))
        llvm_unreachable("Invalid test configuration?");
    }
    return result;
  }
};
} // namespace

/// Returns the Clang resource include directory.
static std::string ResourceInc() {
  llvm::SmallString<256> resource_dir;
  llvm::sys::path::append(resource_dir, GetClangResourceDir().GetPath(),
                          "include");
  return std::string(resource_dir);
}


TEST_F(CppModuleConfigurationTest, Linux) {
  // Test the average Linux configuration.

  std::string usr = "/usr/include";
  std::string libcpp = "/usr/include/c++/v1";
  std::vector<std::string> files = {// C library
                                    usr + "/stdio.h",
                                    // C++ library
                                    libcpp + "/vector",
                                    libcpp + "/module.modulemap"};
  CppModuleConfiguration config(makeFiles(files));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre(libcpp, ResourceInc(), usr));
}

TEST_F(CppModuleConfigurationTest, Sysroot) {
  // Test that having a sysroot for the whole system works fine.

  std::string libcpp = "/home/user/sysroot/usr/include/c++/v1";
  std::string usr = "/home/user/sysroot/usr/include";
  std::vector<std::string> files = {// C library
                                    usr + "/stdio.h",
                                    // C++ library
                                    libcpp + "/vector",
                                    libcpp + "/module.modulemap"};
  CppModuleConfiguration config(makeFiles(files));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre(libcpp, ResourceInc(), usr));
}

TEST_F(CppModuleConfigurationTest, LinuxLocalLibCpp) {
  // Test that a locally build libc++ is detected.

  std::string usr = "/usr/include";
  std::string libcpp = "/home/user/llvm-build/include/c++/v1";
  std::vector<std::string> files = {// C library
                                    usr + "/stdio.h",
                                    // C++ library
                                    libcpp + "/vector",
                                    libcpp + "/module.modulemap"};
  CppModuleConfiguration config(makeFiles(files));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre(libcpp, ResourceInc(), usr));
}

TEST_F(CppModuleConfigurationTest, UnrelatedLibrary) {
  // Test that having an unrelated library in /usr/include doesn't break.

  std::string usr = "/usr/include";
  std::string libcpp = "/home/user/llvm-build/include/c++/v1";
  std::vector<std::string> files = {// C library
                                    usr + "/stdio.h",
                                    // unrelated library
                                    usr + "/boost/vector",
                                    // C++ library
                                    libcpp + "/vector",
                                    libcpp + "/module.modulemap"};
  CppModuleConfiguration config(makeFiles(files));
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
  std::vector<std::string> files = {
      // C library
      usr + "/stdio.h",
      // C++ library
      libcpp + "/vector",
      libcpp + "/module.modulemap",
  };
  CppModuleConfiguration config(makeFiles(files));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre(libcpp, ResourceInc(), usr));
}

TEST_F(CppModuleConfigurationTest, LibCppV2) {
  // Test that a "v2" of libc++ is still correctly detected.

  std::string libcpp = "/usr/include/c++/v2";
  std::vector<std::string> files = {// C library
                                    "/usr/include/stdio.h",
                                    // C++ library
                                    libcpp + "/vector",
                                    libcpp + "/module.modulemap"};
  CppModuleConfiguration config(makeFiles(files));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre("/usr/include/c++/v2", ResourceInc(),
                                   "/usr/include"));
}

TEST_F(CppModuleConfigurationTest, UnknownLibCppFile) {
  // Test that having some unknown file in the libc++ path doesn't break
  // anything.

  std::string libcpp = "/usr/include/c++/v1";
  std::vector<std::string> files = {// C library
                                    "/usr/include/stdio.h",
                                    // C++ library
                                    libcpp + "/non_existing_file",
                                    libcpp + "/module.modulemap",
                                    libcpp + "/vector"};
  CppModuleConfiguration config(makeFiles(files));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre("std"));
  EXPECT_THAT(config.GetIncludeDirs(),
              testing::ElementsAre("/usr/include/c++/v1", ResourceInc(),
                                   "/usr/include"));
}

TEST_F(CppModuleConfigurationTest, MissingUsrInclude) {
  // Test that we don't load 'std' if we can't find the C standard library.

  std::string libcpp = "/usr/include/c++/v1";
  std::vector<std::string> files = {// C++ library
                                    libcpp + "/vector",
                                    libcpp + "/module.modulemap"};
  CppModuleConfiguration config(makeFiles(files));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre());
  EXPECT_THAT(config.GetIncludeDirs(), testing::ElementsAre());
}

TEST_F(CppModuleConfigurationTest, MissingLibCpp) {
  // Test that we don't load 'std' if we don't have a libc++.

  std::string usr = "/usr/include";
  std::vector<std::string> files = {
      // C library
      usr + "/stdio.h",
  };
  CppModuleConfiguration config(makeFiles(files));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre());
  EXPECT_THAT(config.GetIncludeDirs(), testing::ElementsAre());
}

TEST_F(CppModuleConfigurationTest, IgnoreLibStdCpp) {
  // Test that we don't do anything bad when we encounter libstdc++ paths.

  std::string usr = "/usr/include";
  std::vector<std::string> files = {
      // C library
      usr + "/stdio.h",
      // C++ library
      usr + "/c++/8.0.1/vector",
  };
  CppModuleConfiguration config(makeFiles(files));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre());
  EXPECT_THAT(config.GetIncludeDirs(), testing::ElementsAre());
}

TEST_F(CppModuleConfigurationTest, AmbiguousCLib) {
  // Test that we don't do anything when we are not sure where the
  // right C standard library is.

  std::string usr1 = "/usr/include";
  std::string usr2 = "/usr/include/other/path";
  std::string libcpp = usr1 + "c++/v1";
  std::vector<std::string> files = {
      // First C library
      usr1 + "/stdio.h",
      // Second C library
      usr2 + "/stdio.h",
      // C++ library
      libcpp + "/vector",
      libcpp + "/module.modulemap",
  };
  CppModuleConfiguration config(makeFiles(files));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre());
  EXPECT_THAT(config.GetIncludeDirs(), testing::ElementsAre());
}

TEST_F(CppModuleConfigurationTest, AmbiguousLibCpp) {
  // Test that we don't do anything when we are not sure where the
  // right libc++ is.

  std::string usr = "/usr/include";
  std::string libcpp1 = usr + "c++/v1";
  std::string libcpp2 = usr + "c++/v2";
  std::vector<std::string> files = {
      // C library
      usr + "/stdio.h",
      // First C++ library
      libcpp1 + "/vector",
      libcpp1 + "/module.modulemap",
      // Second C++ library
      libcpp2 + "/vector",
      libcpp2 + "/module.modulemap",
  };
  CppModuleConfiguration config(makeFiles(files));
  EXPECT_THAT(config.GetImportedModules(), testing::ElementsAre());
  EXPECT_THAT(config.GetIncludeDirs(), testing::ElementsAre());
}
