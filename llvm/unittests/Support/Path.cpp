//===- llvm/unittest/Support/Path.cpp - Path tests ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Path.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#ifdef LLVM_ON_WIN32
#include <windows.h>
#include <winerror.h>
#endif

#ifdef LLVM_ON_UNIX
#include <sys/stat.h>
#endif

using namespace llvm;
using namespace llvm::sys;

#define ASSERT_NO_ERROR(x)                                                     \
  if (std::error_code ASSERT_NO_ERROR_ec = x) {                                \
    SmallString<128> MessageStorage;                                           \
    raw_svector_ostream Message(MessageStorage);                               \
    Message << #x ": did not return errc::success.\n"                          \
            << "error number: " << ASSERT_NO_ERROR_ec.value() << "\n"          \
            << "error message: " << ASSERT_NO_ERROR_ec.message() << "\n";      \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  } else {                                                                     \
  }

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

  SmallVector<StringRef, 5> ComponentStack;
  for (SmallVector<StringRef, 40>::const_iterator i = paths.begin(),
                                                  e = paths.end();
                                                  i != e;
                                                  ++i) {
    for (sys::path::const_iterator ci = sys::path::begin(*i),
                                   ce = sys::path::end(*i);
                                   ci != ce;
                                   ++ci) {
      ASSERT_FALSE(ci->empty());
      ComponentStack.push_back(*ci);
    }

    for (sys::path::reverse_iterator ci = sys::path::rbegin(*i),
                                     ce = sys::path::rend(*i);
                                     ci != ce;
                                     ++ci) {
      ASSERT_TRUE(*ci == ComponentStack.back());
      ComponentStack.pop_back();
    }
    ASSERT_TRUE(ComponentStack.empty());

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
    EXPECT_EQ(*sys::path::rbegin(filename), (stem + ext).str());

    path::native(*i, temp_store);
  }

  SmallString<32> Relative("foo.cpp");
  ASSERT_NO_ERROR(sys::fs::make_absolute("/root", Relative));
  Relative[5] = '/'; // Fix up windows paths.
  ASSERT_EQ("/root/foo.cpp", Relative);
}

TEST(Support, RelativePathIterator) {
  SmallString<64> Path(StringRef("c/d/e/foo.txt"));
  typedef SmallVector<StringRef, 4> PathComponents;
  PathComponents ExpectedPathComponents;
  PathComponents ActualPathComponents;

  StringRef(Path).split(ExpectedPathComponents, '/');

  for (path::const_iterator I = path::begin(Path), E = path::end(Path); I != E;
       ++I) {
    ActualPathComponents.push_back(*I);
  }

  ASSERT_EQ(ExpectedPathComponents.size(), ActualPathComponents.size());

  for (size_t i = 0; i <ExpectedPathComponents.size(); ++i) {
    EXPECT_EQ(ExpectedPathComponents[i].str(), ActualPathComponents[i].str());
  }
}

TEST(Support, RelativePathDotIterator) {
  SmallString<64> Path(StringRef(".c/.d/../."));
  typedef SmallVector<StringRef, 4> PathComponents;
  PathComponents ExpectedPathComponents;
  PathComponents ActualPathComponents;

  StringRef(Path).split(ExpectedPathComponents, '/');

  for (path::const_iterator I = path::begin(Path), E = path::end(Path); I != E;
       ++I) {
    ActualPathComponents.push_back(*I);
  }

  ASSERT_EQ(ExpectedPathComponents.size(), ActualPathComponents.size());

  for (size_t i = 0; i <ExpectedPathComponents.size(); ++i) {
    EXPECT_EQ(ExpectedPathComponents[i].str(), ActualPathComponents[i].str());
  }
}

TEST(Support, AbsolutePathIterator) {
  SmallString<64> Path(StringRef("/c/d/e/foo.txt"));
  typedef SmallVector<StringRef, 4> PathComponents;
  PathComponents ExpectedPathComponents;
  PathComponents ActualPathComponents;

  StringRef(Path).split(ExpectedPathComponents, '/');

  // The root path will also be a component when iterating
  ExpectedPathComponents[0] = "/";

  for (path::const_iterator I = path::begin(Path), E = path::end(Path); I != E;
       ++I) {
    ActualPathComponents.push_back(*I);
  }

  ASSERT_EQ(ExpectedPathComponents.size(), ActualPathComponents.size());

  for (size_t i = 0; i <ExpectedPathComponents.size(); ++i) {
    EXPECT_EQ(ExpectedPathComponents[i].str(), ActualPathComponents[i].str());
  }
}

TEST(Support, AbsolutePathDotIterator) {
  SmallString<64> Path(StringRef("/.c/.d/../."));
  typedef SmallVector<StringRef, 4> PathComponents;
  PathComponents ExpectedPathComponents;
  PathComponents ActualPathComponents;

  StringRef(Path).split(ExpectedPathComponents, '/');

  // The root path will also be a component when iterating
  ExpectedPathComponents[0] = "/";

  for (path::const_iterator I = path::begin(Path), E = path::end(Path); I != E;
       ++I) {
    ActualPathComponents.push_back(*I);
  }

  ASSERT_EQ(ExpectedPathComponents.size(), ActualPathComponents.size());

  for (size_t i = 0; i <ExpectedPathComponents.size(); ++i) {
    EXPECT_EQ(ExpectedPathComponents[i].str(), ActualPathComponents[i].str());
  }
}

#ifdef LLVM_ON_WIN32
TEST(Support, AbsolutePathIteratorWin32) {
  SmallString<64> Path(StringRef("c:\\c\\e\\foo.txt"));
  typedef SmallVector<StringRef, 4> PathComponents;
  PathComponents ExpectedPathComponents;
  PathComponents ActualPathComponents;

  StringRef(Path).split(ExpectedPathComponents, "\\");

  // The root path (which comes after the drive name) will also be a component
  // when iterating.
  ExpectedPathComponents.insert(ExpectedPathComponents.begin()+1, "\\");

  for (path::const_iterator I = path::begin(Path), E = path::end(Path); I != E;
       ++I) {
    ActualPathComponents.push_back(*I);
  }

  ASSERT_EQ(ExpectedPathComponents.size(), ActualPathComponents.size());

  for (size_t i = 0; i <ExpectedPathComponents.size(); ++i) {
    EXPECT_EQ(ExpectedPathComponents[i].str(), ActualPathComponents[i].str());
  }
}
#endif // LLVM_ON_WIN32

TEST(Support, AbsolutePathIteratorEnd) {
  // Trailing slashes are converted to '.' unless they are part of the root path.
  SmallVector<StringRef, 4> Paths;
  Paths.push_back("/foo/");
  Paths.push_back("/foo//");
  Paths.push_back("//net//");
#ifdef LLVM_ON_WIN32
  Paths.push_back("c:\\\\");
#endif

  for (StringRef Path : Paths) {
    StringRef LastComponent = *path::rbegin(Path);
    EXPECT_EQ(".", LastComponent);
  }

  SmallVector<StringRef, 3> RootPaths;
  RootPaths.push_back("/");
  RootPaths.push_back("//net/");
#ifdef LLVM_ON_WIN32
  RootPaths.push_back("c:\\");
#endif

  for (StringRef Path : RootPaths) {
    StringRef LastComponent = *path::rbegin(Path);
    EXPECT_EQ(1u, LastComponent.size());
    EXPECT_TRUE(path::is_separator(LastComponent[0]));
  }
}

TEST(Support, HomeDirectory) {
  std::string expected;
#ifdef LLVM_ON_WIN32
  if (wchar_t const *path = ::_wgetenv(L"USERPROFILE")) {
    auto pathLen = ::wcslen(path);
    ArrayRef<char> ref{reinterpret_cast<char const *>(path),
                       pathLen * sizeof(wchar_t)};
    convertUTF16ToUTF8String(ref, expected);
  }
#else
  if (char const *path = ::getenv("HOME"))
    expected = path;
#endif
  // Do not try to test it if we don't know what to expect.
  // On Windows we use something better than env vars.
  if (!expected.empty()) {
    SmallString<128> HomeDir;
    auto status = path::home_directory(HomeDir);
    EXPECT_TRUE(status);
    EXPECT_EQ(expected, HomeDir);
  }
}

TEST(Support, UserCacheDirectory) {
  SmallString<13> CacheDir;
  SmallString<20> CacheDir2;
  auto Status = path::user_cache_directory(CacheDir, "");
  EXPECT_TRUE(Status ^ CacheDir.empty());

  if (Status) {
    EXPECT_TRUE(path::user_cache_directory(CacheDir2, "")); // should succeed
    EXPECT_EQ(CacheDir, CacheDir2); // and return same paths

    EXPECT_TRUE(path::user_cache_directory(CacheDir, "A", "B", "file.c"));
    auto It = path::rbegin(CacheDir);
    EXPECT_EQ("file.c", *It);
    EXPECT_EQ("B", *++It);
    EXPECT_EQ("A", *++It);
    auto ParentDir = *++It;

    // Test Unicode: "<user_cache_dir>/(pi)r^2/aleth.0"
    EXPECT_TRUE(path::user_cache_directory(CacheDir2, "\xCF\x80r\xC2\xB2",
                                           "\xE2\x84\xB5.0"));
    auto It2 = path::rbegin(CacheDir2);
    EXPECT_EQ("\xE2\x84\xB5.0", *It2);
    EXPECT_EQ("\xCF\x80r\xC2\xB2", *++It2);
    auto ParentDir2 = *++It2;

    EXPECT_EQ(ParentDir, ParentDir2);
  }
}

class FileSystemTest : public testing::Test {
protected:
  /// Unique temporary directory in which all created filesystem entities must
  /// be placed. It is removed at the end of each test (must be empty).
  SmallString<128> TestDirectory;

  void SetUp() override {
    ASSERT_NO_ERROR(
        fs::createUniqueDirectory("file-system-test", TestDirectory));
    // We don't care about this specific file.
    errs() << "Test Directory: " << TestDirectory << '\n';
    errs().flush();
  }

  void TearDown() override { ASSERT_NO_ERROR(fs::remove(TestDirectory.str())); }
};

TEST_F(FileSystemTest, Unique) {
  // Create a temp file.
  int FileDescriptor;
  SmallString<64> TempPath;
  ASSERT_NO_ERROR(
      fs::createTemporaryFile("prefix", "temp", FileDescriptor, TempPath));

  // The same file should return an identical unique id.
  fs::UniqueID F1, F2;
  ASSERT_NO_ERROR(fs::getUniqueID(Twine(TempPath), F1));
  ASSERT_NO_ERROR(fs::getUniqueID(Twine(TempPath), F2));
  ASSERT_EQ(F1, F2);

  // Different files should return different unique ids.
  int FileDescriptor2;
  SmallString<64> TempPath2;
  ASSERT_NO_ERROR(
      fs::createTemporaryFile("prefix", "temp", FileDescriptor2, TempPath2));

  fs::UniqueID D;
  ASSERT_NO_ERROR(fs::getUniqueID(Twine(TempPath2), D));
  ASSERT_NE(D, F1);
  ::close(FileDescriptor2);

  ASSERT_NO_ERROR(fs::remove(Twine(TempPath2)));

  // Two paths representing the same file on disk should still provide the
  // same unique id.  We can test this by making a hard link.
  ASSERT_NO_ERROR(fs::create_link(Twine(TempPath), Twine(TempPath2)));
  fs::UniqueID D2;
  ASSERT_NO_ERROR(fs::getUniqueID(Twine(TempPath2), D2));
  ASSERT_EQ(D2, F1);

  ::close(FileDescriptor);

  SmallString<128> Dir1;
  ASSERT_NO_ERROR(
     fs::createUniqueDirectory("dir1", Dir1));
  ASSERT_NO_ERROR(fs::getUniqueID(Dir1.c_str(), F1));
  ASSERT_NO_ERROR(fs::getUniqueID(Dir1.c_str(), F2));
  ASSERT_EQ(F1, F2);

  SmallString<128> Dir2;
  ASSERT_NO_ERROR(
     fs::createUniqueDirectory("dir2", Dir2));
  ASSERT_NO_ERROR(fs::getUniqueID(Dir2.c_str(), F2));
  ASSERT_NE(F1, F2);
}

TEST_F(FileSystemTest, TempFiles) {
  // Create a temp file.
  int FileDescriptor;
  SmallString<64> TempPath;
  ASSERT_NO_ERROR(
      fs::createTemporaryFile("prefix", "temp", FileDescriptor, TempPath));

  // Make sure it exists.
  ASSERT_TRUE(sys::fs::exists(Twine(TempPath)));

  // Create another temp tile.
  int FD2;
  SmallString<64> TempPath2;
  ASSERT_NO_ERROR(fs::createTemporaryFile("prefix", "temp", FD2, TempPath2));
  ASSERT_TRUE(TempPath2.endswith(".temp"));
  ASSERT_NE(TempPath.str(), TempPath2.str());

  fs::file_status A, B;
  ASSERT_NO_ERROR(fs::status(Twine(TempPath), A));
  ASSERT_NO_ERROR(fs::status(Twine(TempPath2), B));
  EXPECT_FALSE(fs::equivalent(A, B));

  ::close(FD2);

  // Remove Temp2.
  ASSERT_NO_ERROR(fs::remove(Twine(TempPath2)));
  ASSERT_NO_ERROR(fs::remove(Twine(TempPath2)));
  ASSERT_EQ(fs::remove(Twine(TempPath2), false),
            errc::no_such_file_or_directory);

  std::error_code EC = fs::status(TempPath2.c_str(), B);
  EXPECT_EQ(EC, errc::no_such_file_or_directory);
  EXPECT_EQ(B.type(), fs::file_type::file_not_found);

  // Make sure Temp2 doesn't exist.
  ASSERT_EQ(fs::access(Twine(TempPath2), sys::fs::AccessMode::Exist),
            errc::no_such_file_or_directory);

  SmallString<64> TempPath3;
  ASSERT_NO_ERROR(fs::createTemporaryFile("prefix", "", TempPath3));
  ASSERT_FALSE(TempPath3.endswith("."));

  // Create a hard link to Temp1.
  ASSERT_NO_ERROR(fs::create_link(Twine(TempPath), Twine(TempPath2)));
  bool equal;
  ASSERT_NO_ERROR(fs::equivalent(Twine(TempPath), Twine(TempPath2), equal));
  EXPECT_TRUE(equal);
  ASSERT_NO_ERROR(fs::status(Twine(TempPath), A));
  ASSERT_NO_ERROR(fs::status(Twine(TempPath2), B));
  EXPECT_TRUE(fs::equivalent(A, B));

  // Remove Temp1.
  ::close(FileDescriptor);
  ASSERT_NO_ERROR(fs::remove(Twine(TempPath)));

  // Remove the hard link.
  ASSERT_NO_ERROR(fs::remove(Twine(TempPath2)));

  // Make sure Temp1 doesn't exist.
  ASSERT_EQ(fs::access(Twine(TempPath), sys::fs::AccessMode::Exist),
            errc::no_such_file_or_directory);

#ifdef LLVM_ON_WIN32
  // Path name > 260 chars should get an error.
  const char *Path270 =
    "abcdefghijklmnopqrstuvwxyz9abcdefghijklmnopqrstuvwxyz8"
    "abcdefghijklmnopqrstuvwxyz7abcdefghijklmnopqrstuvwxyz6"
    "abcdefghijklmnopqrstuvwxyz5abcdefghijklmnopqrstuvwxyz4"
    "abcdefghijklmnopqrstuvwxyz3abcdefghijklmnopqrstuvwxyz2"
    "abcdefghijklmnopqrstuvwxyz1abcdefghijklmnopqrstuvwxyz0";
  EXPECT_EQ(fs::createUniqueFile(Path270, FileDescriptor, TempPath),
            errc::invalid_argument);
  // Relative path < 247 chars, no problem.
  const char *Path216 =
    "abcdefghijklmnopqrstuvwxyz7abcdefghijklmnopqrstuvwxyz6"
    "abcdefghijklmnopqrstuvwxyz5abcdefghijklmnopqrstuvwxyz4"
    "abcdefghijklmnopqrstuvwxyz3abcdefghijklmnopqrstuvwxyz2"
    "abcdefghijklmnopqrstuvwxyz1abcdefghijklmnopqrstuvwxyz0";
  ASSERT_NO_ERROR(fs::createTemporaryFile(Path216, "", TempPath));
  ASSERT_NO_ERROR(fs::remove(Twine(TempPath)));
#endif
}

TEST_F(FileSystemTest, CreateDir) {
  ASSERT_NO_ERROR(fs::create_directory(Twine(TestDirectory) + "foo"));
  ASSERT_NO_ERROR(fs::create_directory(Twine(TestDirectory) + "foo"));
  ASSERT_EQ(fs::create_directory(Twine(TestDirectory) + "foo", false),
            errc::file_exists);
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "foo"));

#ifdef LLVM_ON_UNIX
  // Set a 0000 umask so that we can test our directory permissions.
  mode_t OldUmask = ::umask(0000);

  fs::file_status Status;
  ASSERT_NO_ERROR(
      fs::create_directory(Twine(TestDirectory) + "baz500", false,
                           fs::perms::owner_read | fs::perms::owner_exe));
  ASSERT_NO_ERROR(fs::status(Twine(TestDirectory) + "baz500", Status));
  ASSERT_EQ(Status.permissions() & fs::perms::all_all,
            fs::perms::owner_read | fs::perms::owner_exe);
  ASSERT_NO_ERROR(fs::create_directory(Twine(TestDirectory) + "baz777", false,
                                       fs::perms::all_all));
  ASSERT_NO_ERROR(fs::status(Twine(TestDirectory) + "baz777", Status));
  ASSERT_EQ(Status.permissions() & fs::perms::all_all, fs::perms::all_all);

  // Restore umask to be safe.
  ::umask(OldUmask);
#endif

#ifdef LLVM_ON_WIN32
  // Prove that create_directories() can handle a pathname > 248 characters,
  // which is the documented limit for CreateDirectory().
  // (248 is MAX_PATH subtracting room for an 8.3 filename.)
  // Generate a directory path guaranteed to fall into that range.
  size_t TmpLen = TestDirectory.size();
  const char *OneDir = "\\123456789";
  size_t OneDirLen = strlen(OneDir);
  ASSERT_LT(OneDirLen, 12U);
  size_t NLevels = ((248 - TmpLen) / OneDirLen) + 1;
  SmallString<260> LongDir(TestDirectory);
  for (size_t I = 0; I < NLevels; ++I)
    LongDir.append(OneDir);
  ASSERT_NO_ERROR(fs::create_directories(Twine(LongDir)));
  ASSERT_NO_ERROR(fs::create_directories(Twine(LongDir)));
  ASSERT_EQ(fs::create_directories(Twine(LongDir), false),
            errc::file_exists);
  // Tidy up, "recursively" removing the directories.
  StringRef ThisDir(LongDir);
  for (size_t J = 0; J < NLevels; ++J) {
    ASSERT_NO_ERROR(fs::remove(ThisDir));
    ThisDir = path::parent_path(ThisDir);
  }

  // Similarly for a relative pathname.  Need to set the current directory to
  // TestDirectory so that the one we create ends up in the right place.
  char PreviousDir[260];
  size_t PreviousDirLen = ::GetCurrentDirectoryA(260, PreviousDir);
  ASSERT_GT(PreviousDirLen, 0U);
  ASSERT_LT(PreviousDirLen, 260U);
  ASSERT_NE(::SetCurrentDirectoryA(TestDirectory.c_str()), 0);
  LongDir.clear();
  // Generate a relative directory name with absolute length > 248.
  size_t LongDirLen = 249 - TestDirectory.size();
  LongDir.assign(LongDirLen, 'a');
  ASSERT_NO_ERROR(fs::create_directory(Twine(LongDir)));
  // While we're here, prove that .. and . handling works in these long paths.
  const char *DotDotDirs = "\\..\\.\\b";
  LongDir.append(DotDotDirs);
  ASSERT_NO_ERROR(fs::create_directory("b"));
  ASSERT_EQ(fs::create_directory(Twine(LongDir), false), errc::file_exists);
  // And clean up.
  ASSERT_NO_ERROR(fs::remove("b"));
  ASSERT_NO_ERROR(fs::remove(
    Twine(LongDir.substr(0, LongDir.size() - strlen(DotDotDirs)))));
  ASSERT_NE(::SetCurrentDirectoryA(PreviousDir), 0);
#endif
}

TEST_F(FileSystemTest, DirectoryIteration) {
  std::error_code ec;
  for (fs::directory_iterator i(".", ec), e; i != e; i.increment(ec))
    ASSERT_NO_ERROR(ec);

  // Create a known hierarchy to recurse over.
  ASSERT_NO_ERROR(
      fs::create_directories(Twine(TestDirectory) + "/recursive/a0/aa1"));
  ASSERT_NO_ERROR(
      fs::create_directories(Twine(TestDirectory) + "/recursive/a0/ab1"));
  ASSERT_NO_ERROR(fs::create_directories(Twine(TestDirectory) +
                                         "/recursive/dontlookhere/da1"));
  ASSERT_NO_ERROR(
      fs::create_directories(Twine(TestDirectory) + "/recursive/z0/za1"));
  ASSERT_NO_ERROR(
      fs::create_directories(Twine(TestDirectory) + "/recursive/pop/p1"));
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

  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/recursive/a0/aa1"));
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/recursive/a0/ab1"));
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/recursive/a0"));
  ASSERT_NO_ERROR(
      fs::remove(Twine(TestDirectory) + "/recursive/dontlookhere/da1"));
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/recursive/dontlookhere"));
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/recursive/pop/p1"));
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/recursive/pop"));
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/recursive/z0/za1"));
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/recursive/z0"));
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/recursive"));
}

const char archive[] = "!<arch>\x0A";
const char bitcode[] = "\xde\xc0\x17\x0b";
const char coff_object[] = "\x00\x00......";
const char coff_bigobj[] = "\x00\x00\xff\xff\x00\x02......"
    "\xc7\xa1\xba\xd1\xee\xba\xa9\x4b\xaf\x20\xfa\xf6\x6a\xa4\xdc\xb8";
const char coff_import_library[] = "\x00\x00\xff\xff....";
const char elf_relocatable[] = { 0x7f, 'E', 'L', 'F', 1, 2, 1, 0, 0,
                                 0,    0,   0,   0,   0, 0, 0, 0, 1 };
const char macho_universal_binary[] = "\xca\xfe\xba\xbe...\0x00";
const char macho_object[] = "\xfe\xed\xfa\xce..........\x00\x01";
const char macho_executable[] = "\xfe\xed\xfa\xce..........\x00\x02";
const char macho_fixed_virtual_memory_shared_lib[] =
    "\xfe\xed\xfa\xce..........\x00\x03";
const char macho_core[] = "\xfe\xed\xfa\xce..........\x00\x04";
const char macho_preload_executable[] = "\xfe\xed\xfa\xce..........\x00\x05";
const char macho_dynamically_linked_shared_lib[] =
    "\xfe\xed\xfa\xce..........\x00\x06";
const char macho_dynamic_linker[] = "\xfe\xed\xfa\xce..........\x00\x07";
const char macho_bundle[] = "\xfe\xed\xfa\xce..........\x00\x08";
const char macho_dsym_companion[] = "\xfe\xed\xfa\xce..........\x00\x0a";
const char macho_kext_bundle[] = "\xfe\xed\xfa\xce..........\x00\x0b";
const char windows_resource[] = "\x00\x00\x00\x00\x020\x00\x00\x00\xff";
const char macho_dynamically_linked_shared_lib_stub[] =
    "\xfe\xed\xfa\xce..........\x00\x09";

TEST_F(FileSystemTest, Magic) {
  struct type {
    const char *filename;
    const char *magic_str;
    size_t magic_str_len;
    fs::file_magic magic;
  } types[] = {
#define DEFINE(magic)                                           \
    { #magic, magic, sizeof(magic), fs::file_magic::magic }
    DEFINE(archive),
    DEFINE(bitcode),
    DEFINE(coff_object),
    { "coff_bigobj", coff_bigobj, sizeof(coff_bigobj), fs::file_magic::coff_object },
    DEFINE(coff_import_library),
    DEFINE(elf_relocatable),
    DEFINE(macho_universal_binary),
    DEFINE(macho_object),
    DEFINE(macho_executable),
    DEFINE(macho_fixed_virtual_memory_shared_lib),
    DEFINE(macho_core),
    DEFINE(macho_preload_executable),
    DEFINE(macho_dynamically_linked_shared_lib),
    DEFINE(macho_dynamic_linker),
    DEFINE(macho_bundle),
    DEFINE(macho_dynamically_linked_shared_lib_stub),
    DEFINE(macho_dsym_companion),
    DEFINE(macho_kext_bundle),
    DEFINE(windows_resource)
#undef DEFINE
    };

  // Create some files filled with magic.
  for (type *i = types, *e = types + (sizeof(types) / sizeof(type)); i != e;
                                                                     ++i) {
    SmallString<128> file_pathname(TestDirectory);
    path::append(file_pathname, i->filename);
    std::error_code EC;
    raw_fd_ostream file(file_pathname, EC, sys::fs::F_None);
    ASSERT_FALSE(file.has_error());
    StringRef magic(i->magic_str, i->magic_str_len);
    file << magic;
    file.close();
    EXPECT_EQ(i->magic, fs::identify_magic(magic));
    ASSERT_NO_ERROR(fs::remove(Twine(file_pathname)));
  }
}

#ifdef LLVM_ON_WIN32
TEST_F(FileSystemTest, CarriageReturn) {
  SmallString<128> FilePathname(TestDirectory);
  std::error_code EC;
  path::append(FilePathname, "test");

  {
    raw_fd_ostream File(FilePathname, EC, sys::fs::F_Text);
    ASSERT_NO_ERROR(EC);
    File << '\n';
  }
  {
    auto Buf = MemoryBuffer::getFile(FilePathname.str());
    EXPECT_TRUE((bool)Buf);
    EXPECT_EQ(Buf.get()->getBuffer(), "\r\n");
  }

  {
    raw_fd_ostream File(FilePathname, EC, sys::fs::F_None);
    ASSERT_NO_ERROR(EC);
    File << '\n';
  }
  {
    auto Buf = MemoryBuffer::getFile(FilePathname.str());
    EXPECT_TRUE((bool)Buf);
    EXPECT_EQ(Buf.get()->getBuffer(), "\n");
  }
  ASSERT_NO_ERROR(fs::remove(Twine(FilePathname)));
}
#endif

TEST_F(FileSystemTest, Resize) {
  int FD;
  SmallString<64> TempPath;
  ASSERT_NO_ERROR(fs::createTemporaryFile("prefix", "temp", FD, TempPath));
  ASSERT_NO_ERROR(fs::resize_file(FD, 123));
  fs::file_status Status;
  ASSERT_NO_ERROR(fs::status(FD, Status));
  ASSERT_EQ(Status.getSize(), 123U);
}

TEST_F(FileSystemTest, FileMapping) {
  // Create a temp file.
  int FileDescriptor;
  SmallString<64> TempPath;
  ASSERT_NO_ERROR(
      fs::createTemporaryFile("prefix", "temp", FileDescriptor, TempPath));
  unsigned Size = 4096;
  ASSERT_NO_ERROR(fs::resize_file(FileDescriptor, Size));

  // Map in temp file and add some content
  std::error_code EC;
  StringRef Val("hello there");
  {
    fs::mapped_file_region mfr(FileDescriptor,
                               fs::mapped_file_region::readwrite, Size, 0, EC);
    ASSERT_NO_ERROR(EC);
    std::copy(Val.begin(), Val.end(), mfr.data());
    // Explicitly add a 0.
    mfr.data()[Val.size()] = 0;
    // Unmap temp file
  }

  // Map it back in read-only
  int FD;
  EC = fs::openFileForRead(Twine(TempPath), FD);
  ASSERT_NO_ERROR(EC);
  fs::mapped_file_region mfr(FD, fs::mapped_file_region::readonly, Size, 0, EC);
  ASSERT_NO_ERROR(EC);

  // Verify content
  EXPECT_EQ(StringRef(mfr.const_data()), Val);

  // Unmap temp file
  fs::mapped_file_region m(FD, fs::mapped_file_region::readonly, Size, 0, EC);
  ASSERT_NO_ERROR(EC);
  ASSERT_EQ(close(FD), 0);
}

TEST(Support, NormalizePath) {
#if defined(LLVM_ON_WIN32)
#define EXPECT_PATH_IS(path__, windows__, not_windows__)                        \
  EXPECT_EQ(path__, windows__);
#else
#define EXPECT_PATH_IS(path__, windows__, not_windows__)                        \
  EXPECT_EQ(path__, not_windows__);
#endif

  SmallString<64> Path1("a");
  SmallString<64> Path2("a/b");
  SmallString<64> Path3("a\\b");
  SmallString<64> Path4("a\\\\b");
  SmallString<64> Path5("\\a");
  SmallString<64> Path6("a\\");

  path::native(Path1);
  EXPECT_PATH_IS(Path1, "a", "a");

  path::native(Path2);
  EXPECT_PATH_IS(Path2, "a\\b", "a/b");

  path::native(Path3);
  EXPECT_PATH_IS(Path3, "a\\b", "a/b");

  path::native(Path4);
  EXPECT_PATH_IS(Path4, "a\\\\b", "a\\\\b");

  path::native(Path5);
  EXPECT_PATH_IS(Path5, "\\a", "/a");

  path::native(Path6);
  EXPECT_PATH_IS(Path6, "a\\", "a/");

#undef EXPECT_PATH_IS
}

TEST(Support, RemoveLeadingDotSlash) {
  StringRef Path1("././/foolz/wat");
  StringRef Path2("./////");

  Path1 = path::remove_leading_dotslash(Path1);
  EXPECT_EQ(Path1, "foolz/wat");
  Path2 = path::remove_leading_dotslash(Path2);
  EXPECT_EQ(Path2, "");
}
} // anonymous namespace
