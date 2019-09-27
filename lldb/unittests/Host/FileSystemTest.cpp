//===-- FileSystemTest.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "lldb/Host/FileSystem.h"
#include "llvm/Support/Errc.h"

extern const char *TestMainArgv0;

using namespace lldb_private;
using namespace llvm;
using llvm::sys::fs::UniqueID;

// Modified from llvm/unittests/Support/VirtualFileSystemTest.cpp
namespace {
struct DummyFile : public vfs::File {
  vfs::Status S;
  explicit DummyFile(vfs::Status S) : S(S) {}
  llvm::ErrorOr<vfs::Status> status() override { return S; }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  getBuffer(const Twine &Name, int64_t FileSize, bool RequiresNullTerminator,
            bool IsVolatile) override {
    llvm_unreachable("unimplemented");
  }
  std::error_code close() override { return std::error_code(); }
};

class DummyFileSystem : public vfs::FileSystem {
  int FSID;   // used to produce UniqueIDs
  int FileID; // used to produce UniqueIDs
  std::string cwd;
  std::map<std::string, vfs::Status> FilesAndDirs;

  static int getNextFSID() {
    static int Count = 0;
    return Count++;
  }

public:
  DummyFileSystem() : FSID(getNextFSID()), FileID(0) {}

  ErrorOr<vfs::Status> status(const Twine &Path) override {
    std::map<std::string, vfs::Status>::iterator I =
        FilesAndDirs.find(Path.str());
    if (I == FilesAndDirs.end())
      return make_error_code(llvm::errc::no_such_file_or_directory);
    return I->second;
  }
  ErrorOr<std::unique_ptr<vfs::File>>
  openFileForRead(const Twine &Path) override {
    auto S = status(Path);
    if (S)
      return std::unique_ptr<vfs::File>(new DummyFile{*S});
    return S.getError();
  }
  llvm::ErrorOr<std::string> getCurrentWorkingDirectory() const override {
    return cwd;
  }
  std::error_code setCurrentWorkingDirectory(const Twine &Path) override {
    cwd = Path.str();
    return std::error_code();
  }
  // Map any symlink to "/symlink".
  std::error_code getRealPath(const Twine &Path,
                              SmallVectorImpl<char> &Output) const override {
    auto I = FilesAndDirs.find(Path.str());
    if (I == FilesAndDirs.end())
      return make_error_code(llvm::errc::no_such_file_or_directory);
    if (I->second.isSymlink()) {
      Output.clear();
      Twine("/symlink").toVector(Output);
      return std::error_code();
    }
    Output.clear();
    Path.toVector(Output);
    return std::error_code();
  }

  struct DirIterImpl : public llvm::vfs::detail::DirIterImpl {
    std::map<std::string, vfs::Status> &FilesAndDirs;
    std::map<std::string, vfs::Status>::iterator I;
    std::string Path;
    bool isInPath(StringRef S) {
      if (Path.size() < S.size() && S.find(Path) == 0) {
        auto LastSep = S.find_last_of('/');
        if (LastSep == Path.size() || LastSep == Path.size() - 1)
          return true;
      }
      return false;
    }
    DirIterImpl(std::map<std::string, vfs::Status> &FilesAndDirs,
                const Twine &_Path)
        : FilesAndDirs(FilesAndDirs), I(FilesAndDirs.begin()),
          Path(_Path.str()) {
      for (; I != FilesAndDirs.end(); ++I) {
        if (isInPath(I->first)) {
          CurrentEntry =
              vfs::directory_entry(I->second.getName(), I->second.getType());
          break;
        }
      }
    }
    std::error_code increment() override {
      ++I;
      for (; I != FilesAndDirs.end(); ++I) {
        if (isInPath(I->first)) {
          CurrentEntry =
              vfs::directory_entry(I->second.getName(), I->second.getType());
          break;
        }
      }
      if (I == FilesAndDirs.end())
        CurrentEntry = vfs::directory_entry();
      return std::error_code();
    }
  };

  vfs::directory_iterator dir_begin(const Twine &Dir,
                                    std::error_code &EC) override {
    return vfs::directory_iterator(
        std::make_shared<DirIterImpl>(FilesAndDirs, Dir));
  }

  void addEntry(StringRef Path, const vfs::Status &Status) {
    FilesAndDirs[Path] = Status;
  }

  void addRegularFile(StringRef Path, sys::fs::perms Perms = sys::fs::all_all) {
    vfs::Status S(Path, UniqueID(FSID, FileID++),
                  std::chrono::system_clock::now(), 0, 0, 1024,
                  sys::fs::file_type::regular_file, Perms);
    addEntry(Path, S);
  }

  void addDirectory(StringRef Path, sys::fs::perms Perms = sys::fs::all_all) {
    vfs::Status S(Path, UniqueID(FSID, FileID++),
                  std::chrono::system_clock::now(), 0, 0, 0,
                  sys::fs::file_type::directory_file, Perms);
    addEntry(Path, S);
  }

  void addSymlink(StringRef Path) {
    vfs::Status S(Path, UniqueID(FSID, FileID++),
                  std::chrono::system_clock::now(), 0, 0, 0,
                  sys::fs::file_type::symlink_file, sys::fs::all_all);
    addEntry(Path, S);
  }
};
} // namespace

TEST(FileSystemTest, FileAndDirectoryComponents) {
  using namespace std::chrono;
  FileSystem fs;

#ifdef _WIN32
  FileSpec fs1("C:\\FILE\\THAT\\DOES\\NOT\\EXIST.TXT");
#else
  FileSpec fs1("/file/that/does/not/exist.txt");
#endif
  FileSpec fs2(TestMainArgv0);

  fs.Resolve(fs2);

  EXPECT_EQ(system_clock::time_point(), fs.GetModificationTime(fs1));
  EXPECT_LT(system_clock::time_point() + hours(24 * 365 * 20),
            fs.GetModificationTime(fs2));
}

static IntrusiveRefCntPtr<DummyFileSystem> GetSimpleDummyFS() {
  IntrusiveRefCntPtr<DummyFileSystem> D(new DummyFileSystem());
  D->addRegularFile("/foo");
  D->addDirectory("/bar");
  D->addSymlink("/baz");
  D->addRegularFile("/qux", ~sys::fs::perms::all_read);
  D->setCurrentWorkingDirectory("/");
  return D;
}

TEST(FileSystemTest, Exists) {
  FileSystem fs(GetSimpleDummyFS());

  EXPECT_TRUE(fs.Exists("/foo"));
  EXPECT_TRUE(fs.Exists(FileSpec("/foo", FileSpec::Style::posix)));
}

TEST(FileSystemTest, Readable) {
  FileSystem fs(GetSimpleDummyFS());

  EXPECT_TRUE(fs.Readable("/foo"));
  EXPECT_TRUE(fs.Readable(FileSpec("/foo", FileSpec::Style::posix)));

  EXPECT_FALSE(fs.Readable("/qux"));
  EXPECT_FALSE(fs.Readable(FileSpec("/qux", FileSpec::Style::posix)));
}

TEST(FileSystemTest, GetByteSize) {
  FileSystem fs(GetSimpleDummyFS());

  EXPECT_EQ((uint64_t)1024, fs.GetByteSize("/foo"));
  EXPECT_EQ((uint64_t)1024,
            fs.GetByteSize(FileSpec("/foo", FileSpec::Style::posix)));
}

TEST(FileSystemTest, GetPermissions) {
  FileSystem fs(GetSimpleDummyFS());

  EXPECT_EQ(sys::fs::all_all, fs.GetPermissions("/foo"));
  EXPECT_EQ(sys::fs::all_all,
            fs.GetPermissions(FileSpec("/foo", FileSpec::Style::posix)));
}

TEST(FileSystemTest, MakeAbsolute) {
  FileSystem fs(GetSimpleDummyFS());

  {
    StringRef foo_relative = "foo";
    SmallString<16> foo(foo_relative);
    auto EC = fs.MakeAbsolute(foo);
    EXPECT_FALSE(EC);
    EXPECT_TRUE(foo.equals("/foo"));
  }

  {
    FileSpec file_spec("foo");
    auto EC = fs.MakeAbsolute(file_spec);
    EXPECT_FALSE(EC);
    EXPECT_EQ(FileSpec("/foo"), file_spec);
  }
}

TEST(FileSystemTest, Resolve) {
  FileSystem fs(GetSimpleDummyFS());

  {
    StringRef foo_relative = "foo";
    SmallString<16> foo(foo_relative);
    fs.Resolve(foo);
    EXPECT_TRUE(foo.equals("/foo"));
  }

  {
    FileSpec file_spec("foo");
    fs.Resolve(file_spec);
    EXPECT_EQ(FileSpec("/foo"), file_spec);
  }

  {
    StringRef foo_relative = "bogus";
    SmallString<16> foo(foo_relative);
    fs.Resolve(foo);
    EXPECT_TRUE(foo.equals("bogus"));
  }

  {
    FileSpec file_spec("bogus");
    fs.Resolve(file_spec);
    EXPECT_EQ(FileSpec("bogus"), file_spec);
  }
}

FileSystem::EnumerateDirectoryResult
VFSCallback(void *baton, llvm::sys::fs::file_type file_type,
            llvm::StringRef path) {
  auto visited = static_cast<std::vector<std::string> *>(baton);
  visited->push_back(path.str());
  return FileSystem::eEnumerateDirectoryResultNext;
}

TEST(FileSystemTest, EnumerateDirectory) {
  FileSystem fs(GetSimpleDummyFS());

  std::vector<std::string> visited;

  constexpr bool find_directories = true;
  constexpr bool find_files = true;
  constexpr bool find_other = true;

  fs.EnumerateDirectory("/", find_directories, find_files, find_other,
                        VFSCallback, &visited);

  EXPECT_THAT(visited,
              testing::UnorderedElementsAre("/foo", "/bar", "/baz", "/qux"));
}

TEST(FileSystemTest, OpenErrno) {
#ifdef _WIN32
  FileSpec spec("C:\\FILE\\THAT\\DOES\\NOT\\EXIST.TXT");
#else
  FileSpec spec("/file/that/does/not/exist.txt");
#endif
  FileSystem fs;
  auto file = fs.Open(spec, File::eOpenOptionRead, 0, true);
  ASSERT_FALSE(file);
  std::error_code code = errorToErrorCode(file.takeError());
  EXPECT_EQ(code.category(), std::system_category());
  EXPECT_EQ(code.value(), ENOENT);
}

