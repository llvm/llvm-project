//===- llvm/unittest/Support/Path.cpp - Path tests ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/PathV2.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::sys;

#define ASSERT_NO_ERROR(x) \
  if (error_code ASSERT_NO_ERROR_ec = x) { \
    SmallString<128> MessageStorage; \
    raw_svector_ostream Message(MessageStorage); \
    Message << #x ": did not return errc::success.\n" \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n" \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n"; \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str()); \
  } else {}

namespace {

TEST(is_separator, Works) {
  EXPECT_TRUE(path::is_separator('/'));
  EXPECT_FALSE(path::is_separator('\0'));
  EXPECT_FALSE(path::is_separator('-'));
  EXPECT_FALSE(path::is_separator(' '));

#ifdef LLVM_ON_WIN32
  EXPECT_TRUE(path::is_separator('\\'));
#else
  EXPECT_FALSE(path::is_separator('\\'));
#endif
}

TEST(Support, Path) {
  SmallVector<StringRef, 40> paths;
  paths.push_back("");
  paths.push_back(".");
  paths.push_back("..");
  paths.push_back("foo");
  paths.push_back("/");
  paths.push_back("/foo");
  paths.push_back("foo/");
  paths.push_back("/foo/");
  paths.push_back("foo/bar");
  paths.push_back("/foo/bar");
  paths.push_back("//net");
  paths.push_back("//net/foo");
  paths.push_back("///foo///");
  paths.push_back("///foo///bar");
  paths.push_back("/.");
  paths.push_back("./");
  paths.push_back("/..");
  paths.push_back("../");
  paths.push_back("foo/.");
  paths.push_back("foo/..");
  paths.push_back("foo/./");
  paths.push_back("foo/./bar");
  paths.push_back("foo/..");
  paths.push_back("foo/../");
  paths.push_back("foo/../bar");
  paths.push_back("c:");
  paths.push_back("c:/");
  paths.push_back("c:foo");
  paths.push_back("c:/foo");
  paths.push_back("c:foo/");
  paths.push_back("c:/foo/");
  paths.push_back("c:/foo/bar");
  paths.push_back("prn:");
  paths.push_back("c:\\");
  paths.push_back("c:foo");
  paths.push_back("c:\\foo");
  paths.push_back("c:foo\\");
  paths.push_back("c:\\foo\\");
  paths.push_back("c:\\foo/");
  paths.push_back("c:/foo\\bar");

  for (SmallVector<StringRef, 40>::const_iterator i = paths.begin(),
                                                  e = paths.end();
                                                  i != e;
                                                  ++i) {
    for (sys::path::const_iterator ci = sys::path::begin(*i),
                                   ce = sys::path::end(*i);
                                   ci != ce;
                                   ++ci) {
      ASSERT_FALSE(ci->empty());
    }

#if 0 // Valgrind is whining about this.
    outs() << "    Reverse Iteration: [";
    for (sys::path::reverse_iterator ci = sys::path::rbegin(*i),
                                     ce = sys::path::rend(*i);
                                     ci != ce;
                                     ++ci) {
      outs() << *ci << ',';
    }
    outs() << "]\n";
#endif

    path::has_root_path(*i);
    path::root_path(*i);
    path::has_root_name(*i);
    path::root_name(*i);
    path::has_root_directory(*i);
    path::root_directory(*i);
    path::has_parent_path(*i);
    path::parent_path(*i);
    path::has_filename(*i);
    path::filename(*i);
    path::has_stem(*i);
    path::stem(*i);
    path::has_extension(*i);
    path::extension(*i);
    path::is_absolute(*i);
    path::is_relative(*i);

    SmallString<128> temp_store;
    temp_store = *i;
    ASSERT_NO_ERROR(fs::make_absolute(temp_store));
    temp_store = *i;
    path::remove_filename(temp_store);

    temp_store = *i;
    path::replace_extension(temp_store, "ext");
    StringRef filename(temp_store.begin(), temp_store.size()), stem, ext;
    stem = path::stem(filename);
    ext  = path::extension(filename);
    EXPECT_EQ(*(--sys::path::end(filename)), (stem + ext).str());

    path::native(*i, temp_store);
  }
}

class FileSystemTest : public testing::Test {
protected:
  /// Unique temporary directory in which all created filesystem entities must
  /// be placed. It is recursively removed at the end of each test.
  SmallString<128> TestDirectory;

  virtual void SetUp() {
    int fd;
    ASSERT_NO_ERROR(
      fs::unique_file("file-system-test-%%-%%-%%-%%/test-directory.anchor", fd,
                      TestDirectory));
    // We don't care about this specific file.
    ::close(fd);
    TestDirectory = path::parent_path(TestDirectory);
    errs() << "Test Directory: " << TestDirectory << '\n';
    errs().flush();
  }

  virtual void TearDown() {
    uint32_t removed;
    ASSERT_NO_ERROR(fs::remove_all(TestDirectory.str(), removed));
  }
};

TEST_F(FileSystemTest, TempFiles) {
  // Create a temp file.
  int FileDescriptor;
  SmallString<64> TempPath;
  ASSERT_NO_ERROR(
    fs::unique_file("%%-%%-%%-%%.temp", FileDescriptor, TempPath));

  // Make sure it exists.
  bool TempFileExists;
  ASSERT_NO_ERROR(sys::fs::exists(Twine(TempPath), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  // Create another temp tile.
  int FD2;
  SmallString<64> TempPath2;
  ASSERT_NO_ERROR(fs::unique_file("%%-%%-%%-%%.temp", FD2, TempPath2));
  ASSERT_NE(TempPath.str(), TempPath2.str());

  fs::file_status A, B;
  ASSERT_NO_ERROR(fs::status(Twine(TempPath), A));
  ASSERT_NO_ERROR(fs::status(Twine(TempPath2), B));
  EXPECT_FALSE(fs::equivalent(A, B));

  // Try to copy the first to the second.
  EXPECT_EQ(
    fs::copy_file(Twine(TempPath), Twine(TempPath2)), errc::file_exists);

  ::close(FD2);
  // Try again with the proper options.
  ASSERT_NO_ERROR(fs::copy_file(Twine(TempPath), Twine(TempPath2),
                                fs::copy_option::overwrite_if_exists));
  // Remove Temp2.
  ASSERT_NO_ERROR(fs::remove(Twine(TempPath2), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  // Make sure Temp2 doesn't exist.
  ASSERT_NO_ERROR(fs::exists(Twine(TempPath2), TempFileExists));
  EXPECT_FALSE(TempFileExists);

  // Create a hard link to Temp1.
  ASSERT_NO_ERROR(fs::create_hard_link(Twine(TempPath), Twine(TempPath2)));
  bool equal;
  ASSERT_NO_ERROR(fs::equivalent(Twine(TempPath), Twine(TempPath2), equal));
  EXPECT_TRUE(equal);
  ASSERT_NO_ERROR(fs::status(Twine(TempPath), A));
  ASSERT_NO_ERROR(fs::status(Twine(TempPath2), B));
  EXPECT_TRUE(fs::equivalent(A, B));

  // Remove Temp1.
  ::close(FileDescriptor);
  ASSERT_NO_ERROR(fs::remove(Twine(TempPath), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  // Remove the hard link.
  ASSERT_NO_ERROR(fs::remove(Twine(TempPath2), TempFileExists));
  EXPECT_TRUE(TempFileExists);

  // Make sure Temp1 doesn't exist.
  ASSERT_NO_ERROR(fs::exists(Twine(TempPath), TempFileExists));
  EXPECT_FALSE(TempFileExists);
}

TEST_F(FileSystemTest, DirectoryIteration) {
  error_code ec;
  for (fs::directory_iterator i(".", ec), e; i != e; i.increment(ec))
    ASSERT_NO_ERROR(ec);

  // Create a known hierarchy to recurse over.
  bool existed;
  ASSERT_NO_ERROR(fs::create_directories(Twine(TestDirectory)
                  + "/recursive/a0/aa1", existed));
  ASSERT_NO_ERROR(fs::create_directories(Twine(TestDirectory)
                  + "/recursive/a0/ab1", existed));
  ASSERT_NO_ERROR(fs::create_directories(Twine(TestDirectory)
                  + "/recursive/dontlookhere/da1", existed));
  ASSERT_NO_ERROR(fs::create_directories(Twine(TestDirectory)
                  + "/recursive/z0/za1", existed));
  ASSERT_NO_ERROR(fs::create_directories(Twine(TestDirectory)
                  + "/recursive/pop/p1", existed));
  typedef std::vector<std::string> v_t;
  v_t visited;
  for (fs::recursive_directory_iterator i(Twine(TestDirectory)
         + "/recursive", ec), e; i != e; i.increment(ec)){
    ASSERT_NO_ERROR(ec);
    if (path::filename(i->path()) == "p1") {
      i.pop();
      // FIXME: recursive_directory_iterator should be more robust.
      if (i == e) break;
    }
    if (path::filename(i->path()) == "dontlookhere")
      i.no_push();
    visited.push_back(path::filename(i->path()));
  }
  v_t::const_iterator a0 = std::find(visited.begin(), visited.end(), "a0");
  v_t::const_iterator aa1 = std::find(visited.begin(), visited.end(), "aa1");
  v_t::const_iterator ab1 = std::find(visited.begin(), visited.end(), "ab1");
  v_t::const_iterator dontlookhere = std::find(visited.begin(), visited.end(),
                                               "dontlookhere");
  v_t::const_iterator da1 = std::find(visited.begin(), visited.end(), "da1");
  v_t::const_iterator z0 = std::find(visited.begin(), visited.end(), "z0");
  v_t::const_iterator za1 = std::find(visited.begin(), visited.end(), "za1");
  v_t::const_iterator pop = std::find(visited.begin(), visited.end(), "pop");
  v_t::const_iterator p1 = std::find(visited.begin(), visited.end(), "p1");

  // Make sure that each path was visited correctly.
  ASSERT_NE(a0, visited.end());
  ASSERT_NE(aa1, visited.end());
  ASSERT_NE(ab1, visited.end());
  ASSERT_NE(dontlookhere, visited.end());
  ASSERT_EQ(da1, visited.end()); // Not visited.
  ASSERT_NE(z0, visited.end());
  ASSERT_NE(za1, visited.end());
  ASSERT_NE(pop, visited.end());
  ASSERT_EQ(p1, visited.end()); // Not visited.

  // Make sure that parents were visited before children. No other ordering
  // guarantees can be made across siblings.
  ASSERT_LT(a0, aa1);
  ASSERT_LT(a0, ab1);
  ASSERT_LT(z0, za1);
}

TEST_F(FileSystemTest, Magic) {
  struct type {
    const char *filename;
    const char *magic_str;
    size_t      magic_str_len;
  } types [] = {{"magic.archive", "!<arch>\x0A", 8}};

  // Create some files filled with magic.
  for (type *i = types, *e = types + (sizeof(types) / sizeof(type)); i != e;
                                                                     ++i) {
    SmallString<128> file_pathname(TestDirectory);
    path::append(file_pathname, i->filename);
    std::string ErrMsg;
    raw_fd_ostream file(file_pathname.c_str(), ErrMsg,
                        raw_fd_ostream::F_Binary);
    ASSERT_FALSE(file.has_error());
    StringRef magic(i->magic_str, i->magic_str_len);
    file << magic;
    file.close();
    bool res = false;
    ASSERT_NO_ERROR(fs::has_magic(file_pathname.c_str(), magic, res));
    EXPECT_TRUE(res);
  }
}

#if !defined(_WIN32) // FIXME: Win32 has different permission schema.
TEST_F(FileSystemTest, Permissions) {
  // Create a temp file.
  int FileDescriptor;
  SmallString<64> TempPath;
  ASSERT_NO_ERROR(
    fs::unique_file("%%-%%-%%-%%.temp", FileDescriptor, TempPath));

  // Mark file as read-only
  const fs::perms AllWrite = fs::owner_write|fs::group_write|fs::others_write;
  ASSERT_NO_ERROR(fs::permissions(Twine(TempPath), fs::remove_perms|AllWrite));
 
  // Verify file is read-only
  fs::file_status Status;
  ASSERT_NO_ERROR(fs::status(Twine(TempPath), Status));
  bool AnyWriteBits = (Status.permissions() & AllWrite);
  EXPECT_FALSE(AnyWriteBits);
  
  // Mark file as read-write
  ASSERT_NO_ERROR(fs::permissions(Twine(TempPath), fs::add_perms|AllWrite));
  
  // Verify file is read-write
  ASSERT_NO_ERROR(fs::status(Twine(TempPath), Status));
  AnyWriteBits = (Status.permissions() & AllWrite);
  EXPECT_TRUE(AnyWriteBits);
}
#endif

TEST_F(FileSystemTest, FileMapping) {
  // Create a temp file.
  int FileDescriptor;
  SmallString<64> TempPath;
  ASSERT_NO_ERROR(
    fs::unique_file("%%-%%-%%-%%.temp", FileDescriptor, TempPath));
  // Map in temp file and add some content
  error_code EC;
  StringRef Val("hello there");
  {
    fs::mapped_file_region mfr(FileDescriptor,
                               fs::mapped_file_region::readwrite,
                               4096,
                               0,
                               EC);
    ASSERT_NO_ERROR(EC);
    std::copy(Val.begin(), Val.end(), mfr.data());
    // Explicitly add a 0.
    mfr.data()[Val.size()] = 0;
    // Unmap temp file
  }
  
  // Map it back in read-only
  fs::mapped_file_region mfr(Twine(TempPath),
                             fs::mapped_file_region::readonly,
                             0,
                             0,
                             EC);
  ASSERT_NO_ERROR(EC);
  
  // Verify content
  EXPECT_EQ(StringRef(mfr.const_data()), Val);
  
  // Unmap temp file

#if LLVM_USE_RVALUE_REFERENCES
  fs::mapped_file_region m(Twine(TempPath),
                             fs::mapped_file_region::readonly,
                             0,
                             0,
                             EC);
  ASSERT_NO_ERROR(EC);
  const char *Data = m.const_data();
  fs::mapped_file_region mfrrv(llvm_move(m));
  EXPECT_EQ(mfrrv.const_data(), Data);
#endif
}
} // anonymous namespace
