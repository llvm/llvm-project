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

using namespace llvm;

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

struct ScopedDir {
  SmallString<128> Path;
  ScopedDir(const Twine &Name, bool Unique = false) {
    std::error_code EC;
    if (Unique) {
      EC = llvm::sys::fs::createUniqueDirectory(Name, Path);
    } else {
      Path = Name.str();
      EC = llvm::sys::fs::create_directory(Twine(Path));
    }
    if (EC)
      Path = "";
    EXPECT_FALSE(EC);
    // Ensure the path is the real path so tests can use it to compare against
    // realpath output.
    SmallString<128> RealPath;
    if (!llvm::sys::fs::real_path(Path, RealPath))
      Path.swap(RealPath);
  }
  ~ScopedDir() {
    if (Path != "") {
      EXPECT_FALSE(llvm::sys::fs::remove_directories(Path.str()));
    }
  }
  operator StringRef() { return Path.str(); }
};

struct ScopedLink {
  SmallString<128> Path;
  ScopedLink(const Twine &To, const Twine &From) {
    Path = From.str();
    std::error_code EC = sys::fs::create_link(To, From);
    if (EC)
      Path = "";
    EXPECT_FALSE(EC);
  }
  ~ScopedLink() {
    if (Path != "") {
      EXPECT_FALSE(llvm::sys::fs::remove(Path.str()));
    }
  }
  operator StringRef() { return Path.str(); }
};

struct ScopedFile {
  SmallString<128> Path;
  ScopedFile(const Twine &Name) {
    std::error_code EC;
    EC = llvm::sys::fs::createUniqueFile(Name, Path);
    if (EC)
      Path = "";
    EXPECT_FALSE(EC);
  }
  ~ScopedFile() {
    if (Path != "") {
      EXPECT_FALSE(llvm::sys::fs::remove(Path.str()));
    }
  }
  operator StringRef() { return Path.str(); }
};
} // end anonymous namespace

TEST(FileCollectorTest, addFile) {
  ScopedDir root("add_file_root", true);
  std::string root_fs = std::string(root.Path.str());
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
  ScopedDir file_root("file_root", true);

  llvm::SmallString<128> aaa = file_root.Path;
  llvm::sys::path::append(aaa, "aaa");
  ScopedFile a(aaa.str());

  llvm::SmallString<128> bbb = file_root.Path;
  llvm::sys::path::append(bbb, "bbb");
  ScopedFile b(bbb.str());

  llvm::SmallString<128> ccc = file_root.Path;
  llvm::sys::path::append(ccc, "ccc");
  ScopedFile c(ccc.str());

  std::string root_fs = std::string(file_root.Path.str());
  TestingFileCollector FileCollector(root_fs, root_fs);

  FileCollector.addDirectory(file_root.Path);

  // Make sure the root is correct.
  EXPECT_EQ(FileCollector.Root, root_fs);

  // Make sure we've seen all the added files.
  EXPECT_TRUE(FileCollector.hasSeen(a.Path));
  EXPECT_TRUE(FileCollector.hasSeen(b.Path));
  EXPECT_TRUE(FileCollector.hasSeen(c.Path));

  // Make sure we've only seen the added files.
  llvm::SmallString<128> ddd = file_root.Path;
  llvm::sys::path::append(ddd, "ddd");
  ScopedFile d(ddd.str());
  EXPECT_FALSE(FileCollector.hasSeen(d.Path));
}

TEST(FileCollectorTest, copyFiles) {
  ScopedDir file_root("file_root", true);
  ScopedFile a(file_root + "/aaa");
  ScopedFile b(file_root + "/bbb");
  ScopedFile c(file_root + "/ccc");

  // Create file collector and add files.
  ScopedDir root("copy_files_root", true);
  std::string root_fs = std::string(root.Path.str());
  TestingFileCollector FileCollector(root_fs, root_fs);
  FileCollector.addFile(a.Path);
  FileCollector.addFile(b.Path);
  FileCollector.addFile(c.Path);

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
  ScopedDir file_root("dir_root", true);
  ScopedDir subdir(file_root + "/subdir");
  ScopedDir subdir2(file_root + "/subdir2");
  ScopedFile a(subdir2 + "/a");

  // Create file collector and add files.
  ScopedDir root("copy_files_root", true);
  std::string root_fs = std::string(root.Path.str());
  TestingFileCollector FileCollector(root_fs, root_fs);
  FileCollector.addFile(a.Path);

  // The empty directory isn't seen until we add it.
  EXPECT_TRUE(FileCollector.hasSeen(a.Path));
  EXPECT_FALSE(FileCollector.hasSeen(subdir.Path));

  FileCollector.addFile(subdir.Path);
  EXPECT_TRUE(FileCollector.hasSeen(subdir.Path));

  // Make sure we can construct the directory.
  std::error_code ec = FileCollector.copyFiles(true);
  EXPECT_FALSE(ec);
  bool IsDirectory = false;
  llvm::SmallString<128> SubdirInRoot = root.Path;
  llvm::sys::path::append(SubdirInRoot,
                          llvm::sys::path::relative_path(subdir.Path));
  ec = sys::fs::is_directory(SubdirInRoot, IsDirectory);
  EXPECT_FALSE(ec);
  ASSERT_TRUE(IsDirectory);
}

TEST(FileCollectorTest, recordVFSAccesses) {
  ScopedDir file_root("dir_root", true);
  ScopedDir subdir(file_root + "/subdir");
  ScopedDir subdir2(file_root + "/subdir2");
  ScopedFile a(subdir2 + "/a");
  ScopedFile b(file_root + "/b");
  ScopedDir subdir3(file_root + "/subdir3");
  ScopedFile subdir3a(subdir3 + "/aa");
  ScopedDir subdir3b(subdir3 + "/subdirb");
  {
    ScopedFile subdir3fileremoved(subdir3 + "/removed");
  }

  // Create file collector and add files.
  ScopedDir root("copy_files_root", true);
  std::string root_fs = std::string(root.Path.str());
  auto Collector = std::make_shared<TestingFileCollector>(root_fs, root_fs);
  auto VFS =
      FileCollector::createCollectorVFS(vfs::getRealFileSystem(), Collector);
  VFS->status(a.Path);
  EXPECT_TRUE(Collector->hasSeen(a.Path));

  VFS->openFileForRead(b.Path);
  EXPECT_TRUE(Collector->hasSeen(b.Path));

  VFS->status(subdir.Path);
  EXPECT_TRUE(Collector->hasSeen(subdir.Path));

#ifndef _WIN32
  std::error_code EC;
  auto It = VFS->dir_begin(subdir3.Path, EC);
  EXPECT_FALSE(EC);
  EXPECT_TRUE(Collector->hasSeen(subdir3.Path));
  EXPECT_TRUE(Collector->hasSeen(subdir3a.Path));
  EXPECT_TRUE(Collector->hasSeen(subdir3b.Path));
  std::string RemovedFileName = (Twine(subdir3.Path) + "/removed").str();
  EXPECT_FALSE(Collector->hasSeen(RemovedFileName));
#endif
}

#ifndef _WIN32
TEST(FileCollectorTest, Symlinks) {
  // Root where the original files live.
  ScopedDir file_root("file_root", true);

  // Create some files in the file root.
  ScopedFile a(file_root + "/aaa");
  ScopedFile b(file_root + "/bbb");
  ScopedFile c(file_root + "/ccc");

  // Create a directory foo with file ddd.
  ScopedDir foo(file_root + "/foo");
  ScopedFile d(foo + "/ddd");

  // Create a file eee in the foo's parent directory.
  ScopedFile e(foo + "/../eee");

  // Create a symlink bar pointing to foo.
  ScopedLink symlink(file_root + "/foo", file_root + "/bar");

  // Root where files are copied to.
  ScopedDir reproducer_root("reproducer_root", true);
  std::string root_fs = std::string(reproducer_root.Path.str());
  TestingFileCollector FileCollector(root_fs, root_fs);

  // Add all the files to the collector.
  FileCollector.addFile(a.Path);
  FileCollector.addFile(b.Path);
  FileCollector.addFile(c.Path);
  FileCollector.addFile(d.Path);
  FileCollector.addFile(e.Path);
  FileCollector.addFile(file_root + "/bar/ddd");

  auto mapping = FileCollector.VFSWriter.getMappings();

  {
    // Make sure the common case works.
    std::string vpath = (file_root + "/aaa").str();
    std::string rpath = (reproducer_root.Path + file_root.Path + "/aaa").str();
    printf("%s -> %s\n", vpath.c_str(), rpath.c_str());
    EXPECT_THAT(mapping, testing::Contains(vfs::YAMLVFSEntry(vpath, rpath)));
  }

  {
    // Make sure the virtual path points to the real source path.
    std::string vpath = (file_root + "/bar/ddd").str();
    std::string rpath =
        (reproducer_root.Path + file_root.Path + "/foo/ddd").str();
    printf("%s -> %s\n", vpath.c_str(), rpath.c_str());
    EXPECT_THAT(mapping, testing::Contains(vfs::YAMLVFSEntry(vpath, rpath)));
  }

  {
    // Make sure that .. is removed from the source path.
    std::string vpath = (file_root + "/eee").str();
    std::string rpath = (reproducer_root.Path + file_root.Path + "/eee").str();
    printf("%s -> %s\n", vpath.c_str(), rpath.c_str());
    EXPECT_THAT(mapping, testing::Contains(vfs::YAMLVFSEntry(vpath, rpath)));
  }
}

TEST(FileCollectorTest, recordVFSSymlinkAccesses) {
  ScopedDir file_root("dir_root", true);
  ScopedFile a(file_root + "/a");
  ScopedLink symlink(file_root + "/a", file_root + "/b");

  // Create file collector and add files.
  ScopedDir root("copy_files_root", true);
  std::string root_fs = std::string(root.Path.str());
  auto Collector = std::make_shared<TestingFileCollector>(root_fs, root_fs);
  auto VFS =
      FileCollector::createCollectorVFS(vfs::getRealFileSystem(), Collector);
  SmallString<256> Output;
  VFS->getRealPath(symlink.Path, Output);
  EXPECT_TRUE(Collector->hasSeen(a.Path));
  EXPECT_TRUE(Collector->hasSeen(symlink.Path));
}
#endif
