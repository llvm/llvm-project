//===-- ZipFileResolverTest.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/common/ZipFileResolver.h"
#include "TestingSupport/SubsystemRAII.h"
#include "TestingSupport/TestUtilities.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace llvm;

namespace {
struct ZipFileDesc {
  std::string zip_name;
  std::string lib_path;
  size_t lib_offset;
  size_t lib_size;
};

class ZipFileResolverTest : public ::testing::Test {
  SubsystemRAII<FileSystem> subsystems;
};

std::vector<ZipFileDesc> TestZipFiles() {
  return {
      // Simple uncompressed zip file, making use of extra fields
      {"zip-test.zip", "lib/arm64-v8a/libzip-test.so", 4096, 3600},

      // Simple uncompressed zip file, where the shared library is the last
      // entry, and the entry has no extra field or comment
      {"zip-test-no-extras.zip", "lib/arm64-v8a/libzip-test.so", 140, 3600},
  };
}

std::string TestZipPath(const ZipFileDesc &desc) {
  FileSpec zip_spec(GetInputFilePath(desc.zip_name));
  FileSystem::Instance().Resolve(zip_spec);
  return zip_spec.GetPath();
}
} // namespace

TEST_F(ZipFileResolverTest, ResolveSharedLibraryPathWithNormalFile) {
  const FileSpec file_spec("/system/lib64/libtest.so");

  ZipFileResolver::FileKind file_kind;
  std::string file_path;
  lldb::offset_t file_offset;
  lldb::offset_t file_size;
  ASSERT_TRUE(ZipFileResolver::ResolveSharedLibraryPath(
      file_spec, file_kind, file_path, file_offset, file_size));

  EXPECT_EQ(file_kind, ZipFileResolver::FileKind::eFileKindNormal);
  EXPECT_EQ(file_path, file_spec.GetPath());
  EXPECT_EQ(file_offset, 0UL);
  EXPECT_EQ(file_size, 0UL);
}

TEST_F(ZipFileResolverTest, ResolveSharedLibraryPathWithZipMissing) {
  const std::string zip_path = TestZipPath(TestZipFiles().at(0));
  const FileSpec file_spec(zip_path + "!/lib/arm64-v8a/libmissing.so");

  ZipFileResolver::FileKind file_kind;
  std::string file_path;
  lldb::offset_t file_offset;
  lldb::offset_t file_size;
  ASSERT_FALSE(ZipFileResolver::ResolveSharedLibraryPath(
      file_spec, file_kind, file_path, file_offset, file_size));
}

TEST_F(ZipFileResolverTest, ResolveSharedLibraryPathWithZipExisting) {
  for (const auto &zip_desc : TestZipFiles()) {
    const std::string zip_path = TestZipPath(zip_desc);
    const FileSpec file_spec(zip_path + "!/" + zip_desc.lib_path);

    ZipFileResolver::FileKind file_kind;
    std::string file_path;
    lldb::offset_t file_offset;
    lldb::offset_t file_size;
    ASSERT_TRUE(ZipFileResolver::ResolveSharedLibraryPath(
        file_spec, file_kind, file_path, file_offset, file_size));

    EXPECT_EQ(file_kind, ZipFileResolver::FileKind::eFileKindZip);
    EXPECT_EQ(file_path, zip_path);
    EXPECT_EQ(file_offset, zip_desc.lib_offset);
    EXPECT_EQ(file_size, zip_desc.lib_size);
  }
}
