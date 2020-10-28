//===-- FileCollectorTest.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "llvm/Support/FileCollector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Testing/Support/SupportHelpers.h"

using namespace llvm;
using llvm::unittest::TempDir;
using llvm::unittest::TempFile;
using llvm::unittest::TempLink;

namespace llvm {
namespace vfs {
inline bool operator==(const llvm::vfs::YAMLVFSEntry &LHS,
                       const llvm::vfs::YAMLVFSEntry &RHS) {
  return LHS.VPath == RHS.VPath && LHS.RPath == RHS.RPath;
}
} // namespace vfs
} // namespace llvm

namespace {
class TestingFileCollector : public FileCollector {
public:
  using FileCollector::FileCollector;
  using FileCollector::Root;
  using FileCollector::Seen;
  using FileCollector::SymlinkMap;
  using FileCollector::VFSWriter;

  bool hasSeen(StringRef fs) {
    return Seen.find(fs) != Seen.end();
  }
};

} // end anonymous namespace

TEST(FileCollectorTest, addFile) {
  TempDir root("add_file_root", /*Unique*/ true);
  std::string root_fs(root.path());
  TestingFileCollector FileCollector(root_fs, root_fs);

  FileCollector.addFile("/path/to/a");
  FileCollector.addFile("/path/to/b");
  FileCollector.addFile("/path/to/c");

  // Make sure the root is correct.
  EXPECT_EQ(FileCollector.Root, root_fs);

  // Make sure we've seen all the added files.
  EXPECT_TRUE(FileCollector.hasSeen("/path/to/a"));
  EXPECT_TRUE(FileCollector.hasSeen("/path/to/b"));
  EXPECT_TRUE(FileCollector.hasSeen("/path/to/c"));

  // Make sure we've only seen the added files.
  EXPECT_FALSE(FileCollector.hasSeen("/path/to/d"));
}

TEST(FileCollectorTest, addDirectory) {
  TempDir file_root("file_root", /*Unique*/ true);

  llvm::SmallString<128> aaa(file_root.path());
  llvm::sys::path::append(aaa, "aaa");
  TempFile a(aaa.str());

  llvm::SmallString<128> bbb(file_root.path());
  llvm::sys::path::append(bbb, "bbb");
  TempFile b(bbb.str());

  llvm::SmallString<128> ccc(file_root.path());
  llvm::sys::path::append(ccc, "ccc");
  TempFile c(ccc.str());

  std::string root_fs(file_root.path());
  TestingFileCollector FileCollector(root_fs, root_fs);

  FileCollector.addDirectory(file_root.path());

  // Make sure the root is correct.
  EXPECT_EQ(FileCollector.Root, root_fs);

  // Make sure we've seen all the added files.
  EXPECT_TRUE(FileCollector.hasSeen(a.path()));
  EXPECT_TRUE(FileCollector.hasSeen(b.path()));
  EXPECT_TRUE(FileCollector.hasSeen(c.path()));

  // Make sure we've only seen the added files.
  llvm::SmallString<128> ddd(file_root.path());
  llvm::sys::path::append(ddd, "ddd");
  TempFile d(ddd);
  EXPECT_FALSE(FileCollector.hasSeen(d.path()));
}

TEST(FileCollectorTest, copyFiles) {
  TempDir file_root("file_root", /*Unique*/ true);
  TempFile a(file_root.path("aaa"));
  TempFile b(file_root.path("bbb"));
  TempFile c(file_root.path("ccc"));

  // Create file collector and add files.
  TempDir root("copy_files_root", /*Unique*/ true);
  std::string root_fs(root.path());
  TestingFileCollector FileCollector(root_fs, root_fs);
  FileCollector.addFile(a.path());
  FileCollector.addFile(b.path());
  FileCollector.addFile(c.path());

  // Make sure we can copy the files.
  std::error_code ec = FileCollector.copyFiles(true);
  EXPECT_FALSE(ec);

  // Now add a bogus file and make sure we error out.
  FileCollector.addFile("/some/bogus/file");
  ec = FileCollector.copyFiles(true);
  EXPECT_TRUE(ec);

  // However, if stop_on_error is true the copy should still succeed.
  ec = FileCollector.copyFiles(false);
  EXPECT_FALSE(ec);
}

TEST(FileCollectorTest, recordAndConstructDirectory) {
  TempDir file_root("dir_root", /*Unique*/ true);
  TempDir subdir(file_root.path("subdir"));
  TempDir subdir2(file_root.path("subdir2"));
  TempFile a(subdir2.path("a"));

  // Create file collector and add files.
  TempDir root("copy_files_root", /*Unique*/ true);
  std::string root_fs(root.path());
  TestingFileCollector FileCollector(root_fs, root_fs);
  FileCollector.addFile(a.path());

  // The empty directory isn't seen until we add it.
  EXPECT_TRUE(FileCollector.hasSeen(a.path()));
  EXPECT_FALSE(FileCollector.hasSeen(subdir.path()));

  FileCollector.addFile(subdir.path());
  EXPECT_TRUE(FileCollector.hasSeen(subdir.path()));

  // Make sure we can construct the directory.
  std::error_code ec = FileCollector.copyFiles(true);
  EXPECT_FALSE(ec);
  bool IsDirectory = false;
  llvm::SmallString<128> SubdirInRoot = root.path();
  llvm::sys::path::append(SubdirInRoot,
                          llvm::sys::path::relative_path(subdir.path()));
  ec = sys::fs::is_directory(SubdirInRoot, IsDirectory);
  EXPECT_FALSE(ec);
  ASSERT_TRUE(IsDirectory);
}

TEST(FileCollectorTest, recordVFSAccesses) {
  TempDir file_root("dir_root", /*Unique*/ true);
  TempDir subdir(file_root.path("subdir"));
  TempDir subdir2(file_root.path("subdir2"));
  TempFile a(subdir2.path("a"));
  TempFile b(file_root.path("b"));
  TempDir subdir3(file_root.path("subdir3"));
  TempFile subdir3a(subdir3.path("aa"));
  TempDir subdir3b(subdir3.path("subdirb"));
  { TempFile subdir3fileremoved(subdir3.path("removed")); }

  // Create file collector and add files.
  TempDir root("copy_files_root", /*Unique*/ true);
  std::string root_fs(root.path());
  auto Collector = std::make_shared<TestingFileCollector>(root_fs, root_fs);
  auto VFS =
      FileCollector::createCollectorVFS(vfs::getRealFileSystem(), Collector);
  VFS->status(a.path());
  EXPECT_TRUE(Collector->hasSeen(a.path()));

  VFS->openFileForRead(b.path());
  EXPECT_TRUE(Collector->hasSeen(b.path()));

  VFS->status(subdir.path());
  EXPECT_TRUE(Collector->hasSeen(subdir.path()));

#ifndef _WIN32
  std::error_code EC;
  auto It = VFS->dir_begin(subdir3.path(), EC);
  EXPECT_FALSE(EC);
  EXPECT_TRUE(Collector->hasSeen(subdir3.path()));
  EXPECT_TRUE(Collector->hasSeen(subdir3a.path()));
  EXPECT_TRUE(Collector->hasSeen(subdir3b.path()));
  std::string RemovedFileName((Twine(subdir3.path("removed"))).str());
  EXPECT_FALSE(Collector->hasSeen(RemovedFileName));
#endif
}

#ifndef _WIN32
TEST(FileCollectorTest, Symlinks) {
  // Root where the original files live.
  TempDir file_root("file_root", /*Unique*/ true);

  // Create some files in the file root.
  TempFile a(file_root.path("aaa"));
  TempFile b(file_root.path("bbb"));
  TempFile c(file_root.path("ccc"));

  // Create a directory foo with file ddd.
  TempDir foo(file_root.path("foo"));
  TempFile d(foo.path("ddd"));

  // Create a file eee in the foo's parent directory.
  TempFile e(foo.path("../eee"));

  // Create a symlink bar pointing to foo.
  TempLink symlink(file_root.path("foo"), file_root.path("bar"));

  // Root where files are copied to.
  TempDir reproducer_root("reproducer_root", /*Unique*/ true);
  std::string root_fs(reproducer_root.path());
  TestingFileCollector FileCollector(root_fs, root_fs);

  // Add all the files to the collector.
  FileCollector.addFile(a.path());
  FileCollector.addFile(b.path());
  FileCollector.addFile(c.path());
  FileCollector.addFile(d.path());
  FileCollector.addFile(e.path());
  FileCollector.addFile(file_root.path() + "/bar/ddd");

  auto mapping = FileCollector.VFSWriter.getMappings();

  {
    // Make sure the common case works.
    std::string vpath = (file_root.path() + "/aaa").str();
    std::string rpath =
        (reproducer_root.path() + file_root.path() + "/aaa").str();
    printf("%s -> %s\n", vpath.c_str(), rpath.c_str());
    EXPECT_THAT(mapping, testing::Contains(vfs::YAMLVFSEntry(vpath, rpath)));
  }

  {
    // Make sure the virtual path points to the real source path.
    std::string vpath = (file_root.path() + "/bar/ddd").str();
    std::string rpath =
        (reproducer_root.path() + file_root.path() + "/foo/ddd").str();
    printf("%s -> %s\n", vpath.c_str(), rpath.c_str());
    EXPECT_THAT(mapping, testing::Contains(vfs::YAMLVFSEntry(vpath, rpath)));
  }

  {
    // Make sure that .. is removed from the source path.
    std::string vpath = (file_root.path() + "/eee").str();
    std::string rpath =
        (reproducer_root.path() + file_root.path() + "/eee").str();
    printf("%s -> %s\n", vpath.c_str(), rpath.c_str());
    EXPECT_THAT(mapping, testing::Contains(vfs::YAMLVFSEntry(vpath, rpath)));
  }
}

TEST(FileCollectorTest, recordVFSSymlinkAccesses) {
  TempDir file_root("dir_root", /*Unique*/ true);
  TempFile a(file_root.path("a"));
  TempLink symlink(file_root.path("a"), file_root.path("b"));

  // Create file collector and add files.
  TempDir root("copy_files_root", true);
  std::string root_fs(root.path());
  auto Collector = std::make_shared<TestingFileCollector>(root_fs, root_fs);
  auto VFS =
      FileCollector::createCollectorVFS(vfs::getRealFileSystem(), Collector);
  SmallString<256> Output;
  VFS->getRealPath(symlink.path(), Output);
  EXPECT_TRUE(Collector->hasSeen(a.path()));
  EXPECT_TRUE(Collector->hasSeen(symlink.path()));
}
#endif
