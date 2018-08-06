//===- llvm/unittest/Support/Path.cpp - Path tests ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Path.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#ifdef _WIN32
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Chrono.h"
#include <windows.h>
#include <winerror.h>
#endif

#ifdef LLVM_ON_UNIX
#include <pwd.h>
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

#define ASSERT_ERROR(x)                                                        \
  if (!x) {                                                                    \
    SmallString<128> MessageStorage;                                           \
    raw_svector_ostream Message(MessageStorage);                               \
    Message << #x ": did not return a failure error code.\n";                  \
    GTEST_FATAL_FAILURE_(MessageStorage.c_str());                              \
  }

namespace {

struct FileDescriptorCloser {
  explicit FileDescriptorCloser(int FD) : FD(FD) {}
  ~FileDescriptorCloser() { ::close(FD); }
  int FD;
};

TEST(is_separator, Works) {
  EXPECT_TRUE(path::is_separator('/'));
  EXPECT_FALSE(path::is_separator('\0'));
  EXPECT_FALSE(path::is_separator('-'));
  EXPECT_FALSE(path::is_separator(' '));

  EXPECT_TRUE(path::is_separator('\\', path::Style::windows));
  EXPECT_FALSE(path::is_separator('\\', path::Style::posix));

#ifdef _WIN32
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
  paths.push_back("//net/");
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
    SCOPED_TRACE(*i);
    SmallVector<StringRef, 5> ComponentStack;
    for (sys::path::const_iterator ci = sys::path::begin(*i),
                                   ce = sys::path::end(*i);
                                   ci != ce;
                                   ++ci) {
      EXPECT_FALSE(ci->empty());
      ComponentStack.push_back(*ci);
    }

    SmallVector<StringRef, 5> ReverseComponentStack;
    for (sys::path::reverse_iterator ci = sys::path::rbegin(*i),
                                     ce = sys::path::rend(*i);
                                     ci != ce;
                                     ++ci) {
      EXPECT_FALSE(ci->empty());
      ReverseComponentStack.push_back(*ci);
    }
    std::reverse(ReverseComponentStack.begin(), ReverseComponentStack.end());
    EXPECT_THAT(ComponentStack, testing::ContainerEq(ReverseComponentStack));

    // Crash test most of the API - since we're iterating over all of our paths
    // here there isn't really anything reasonable to assert on in the results.
    (void)path::has_root_path(*i);
    (void)path::root_path(*i);
    (void)path::has_root_name(*i);
    (void)path::root_name(*i);
    (void)path::has_root_directory(*i);
    (void)path::root_directory(*i);
    (void)path::has_parent_path(*i);
    (void)path::parent_path(*i);
    (void)path::has_filename(*i);
    (void)path::filename(*i);
    (void)path::has_stem(*i);
    (void)path::stem(*i);
    (void)path::has_extension(*i);
    (void)path::extension(*i);
    (void)path::is_absolute(*i);
    (void)path::is_relative(*i);

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

TEST(Support, FilenameParent) {
  EXPECT_EQ("/", path::filename("/"));
  EXPECT_EQ("", path::parent_path("/"));

  EXPECT_EQ("\\", path::filename("c:\\", path::Style::windows));
  EXPECT_EQ("c:", path::parent_path("c:\\", path::Style::windows));

  EXPECT_EQ("/", path::filename("///"));
  EXPECT_EQ("", path::parent_path("///"));

  EXPECT_EQ("\\", path::filename("c:\\\\", path::Style::windows));
  EXPECT_EQ("c:", path::parent_path("c:\\\\", path::Style::windows));

  EXPECT_EQ("bar", path::filename("/foo/bar"));
  EXPECT_EQ("/foo", path::parent_path("/foo/bar"));

  EXPECT_EQ("foo", path::filename("/foo"));
  EXPECT_EQ("/", path::parent_path("/foo"));

  EXPECT_EQ("foo", path::filename("foo"));
  EXPECT_EQ("", path::parent_path("foo"));

  EXPECT_EQ(".", path::filename("foo/"));
  EXPECT_EQ("foo", path::parent_path("foo/"));

  EXPECT_EQ("//net", path::filename("//net"));
  EXPECT_EQ("", path::parent_path("//net"));

  EXPECT_EQ("/", path::filename("//net/"));
  EXPECT_EQ("//net", path::parent_path("//net/"));

  EXPECT_EQ("foo", path::filename("//net/foo"));
  EXPECT_EQ("//net/", path::parent_path("//net/foo"));

  // These checks are just to make sure we do something reasonable with the
  // paths below. They are not meant to prescribe the one true interpretation of
  // these paths. Other decompositions (e.g. "//" -> "" + "//") are also
  // possible.
  EXPECT_EQ("/", path::filename("//"));
  EXPECT_EQ("", path::parent_path("//"));

  EXPECT_EQ("\\", path::filename("\\\\", path::Style::windows));
  EXPECT_EQ("", path::parent_path("\\\\", path::Style::windows));

  EXPECT_EQ("\\", path::filename("\\\\\\", path::Style::windows));
  EXPECT_EQ("", path::parent_path("\\\\\\", path::Style::windows));
}

static std::vector<StringRef>
GetComponents(StringRef Path, path::Style S = path::Style::native) {
  return {path::begin(Path, S), path::end(Path)};
}

TEST(Support, PathIterator) {
  EXPECT_THAT(GetComponents("/foo"), testing::ElementsAre("/", "foo"));
  EXPECT_THAT(GetComponents("/"), testing::ElementsAre("/"));
  EXPECT_THAT(GetComponents("//"), testing::ElementsAre("/"));
  EXPECT_THAT(GetComponents("///"), testing::ElementsAre("/"));
  EXPECT_THAT(GetComponents("c/d/e/foo.txt"),
              testing::ElementsAre("c", "d", "e", "foo.txt"));
  EXPECT_THAT(GetComponents(".c/.d/../."),
              testing::ElementsAre(".c", ".d", "..", "."));
  EXPECT_THAT(GetComponents("/c/d/e/foo.txt"),
              testing::ElementsAre("/", "c", "d", "e", "foo.txt"));
  EXPECT_THAT(GetComponents("/.c/.d/../."),
              testing::ElementsAre("/", ".c", ".d", "..", "."));
  EXPECT_THAT(GetComponents("c:\\c\\e\\foo.txt", path::Style::windows),
              testing::ElementsAre("c:", "\\", "c", "e", "foo.txt"));
  EXPECT_THAT(GetComponents("//net/"), testing::ElementsAre("//net", "/"));
  EXPECT_THAT(GetComponents("//net/c/foo.txt"),
              testing::ElementsAre("//net", "/", "c", "foo.txt"));
}

TEST(Support, AbsolutePathIteratorEnd) {
  // Trailing slashes are converted to '.' unless they are part of the root path.
  SmallVector<std::pair<StringRef, path::Style>, 4> Paths;
  Paths.emplace_back("/foo/", path::Style::native);
  Paths.emplace_back("/foo//", path::Style::native);
  Paths.emplace_back("//net/foo/", path::Style::native);
  Paths.emplace_back("c:\\foo\\", path::Style::windows);

  for (auto &Path : Paths) {
    SCOPED_TRACE(Path.first);
    StringRef LastComponent = *path::rbegin(Path.first, Path.second);
    EXPECT_EQ(".", LastComponent);
  }

  SmallVector<std::pair<StringRef, path::Style>, 3> RootPaths;
  RootPaths.emplace_back("/", path::Style::native);
  RootPaths.emplace_back("//net/", path::Style::native);
  RootPaths.emplace_back("c:\\", path::Style::windows);
  RootPaths.emplace_back("//net//", path::Style::native);
  RootPaths.emplace_back("c:\\\\", path::Style::windows);

  for (auto &Path : RootPaths) {
    SCOPED_TRACE(Path.first);
    StringRef LastComponent = *path::rbegin(Path.first, Path.second);
    EXPECT_EQ(1u, LastComponent.size());
    EXPECT_TRUE(path::is_separator(LastComponent[0], Path.second));
  }
}

TEST(Support, HomeDirectory) {
  std::string expected;
#ifdef _WIN32
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

#ifdef LLVM_ON_UNIX
TEST(Support, HomeDirectoryWithNoEnv) {
  std::string OriginalStorage;
  char const *OriginalEnv = ::getenv("HOME");
  if (OriginalEnv) {
    // We're going to unset it, so make a copy and save a pointer to the copy
    // so that we can reset it at the end of the test.
    OriginalStorage = OriginalEnv;
    OriginalEnv = OriginalStorage.c_str();
  }

  // Don't run the test if we have nothing to compare against.
  struct passwd *pw = getpwuid(getuid());
  if (!pw || !pw->pw_dir) return;

  ::unsetenv("HOME");
  EXPECT_EQ(nullptr, ::getenv("HOME"));
  std::string PwDir = pw->pw_dir;

  SmallString<128> HomeDir;
  auto status = path::home_directory(HomeDir);
  EXPECT_TRUE(status);
  EXPECT_EQ(PwDir, HomeDir);

  // Now put the environment back to its original state (meaning that if it was
  // unset before, we don't reset it).
  if (OriginalEnv) ::setenv("HOME", OriginalEnv, 1);
}
#endif

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

TEST(Support, TempDirectory) {
  SmallString<32> TempDir;
  path::system_temp_directory(false, TempDir);
  EXPECT_TRUE(!TempDir.empty());
  TempDir.clear();
  path::system_temp_directory(true, TempDir);
  EXPECT_TRUE(!TempDir.empty());
}

#ifdef _WIN32
static std::string path2regex(std::string Path) {
  size_t Pos = 0;
  while ((Pos = Path.find('\\', Pos)) != std::string::npos) {
    Path.replace(Pos, 1, "\\\\");
    Pos += 2;
  }
  return Path;
}

/// Helper for running temp dir test in separated process. See below.
#define EXPECT_TEMP_DIR(prepare, expected)                                     \
  EXPECT_EXIT(                                                                 \
      {                                                                        \
        prepare;                                                               \
        SmallString<300> TempDir;                                              \
        path::system_temp_directory(true, TempDir);                            \
        raw_os_ostream(std::cerr) << TempDir;                                  \
        std::exit(0);                                                          \
      },                                                                       \
      ::testing::ExitedWithCode(0), path2regex(expected))

TEST(SupportDeathTest, TempDirectoryOnWindows) {
  // In this test we want to check how system_temp_directory responds to
  // different values of specific env vars. To prevent corrupting env vars of
  // the current process all checks are done in separated processes.
  EXPECT_TEMP_DIR(_wputenv_s(L"TMP", L"C:\\OtherFolder"), "C:\\OtherFolder");
  EXPECT_TEMP_DIR(_wputenv_s(L"TMP", L"C:/Unix/Path/Seperators"),
                  "C:\\Unix\\Path\\Seperators");
  EXPECT_TEMP_DIR(_wputenv_s(L"TMP", L"Local Path"), ".+\\Local Path$");
  EXPECT_TEMP_DIR(_wputenv_s(L"TMP", L"F:\\TrailingSep\\"), "F:\\TrailingSep");
  EXPECT_TEMP_DIR(
      _wputenv_s(L"TMP", L"C:\\2\x03C0r-\x00B5\x00B3\\\x2135\x2080"),
      "C:\\2\xCF\x80r-\xC2\xB5\xC2\xB3\\\xE2\x84\xB5\xE2\x82\x80");

  // Test $TMP empty, $TEMP set.
  EXPECT_TEMP_DIR(
      {
        _wputenv_s(L"TMP", L"");
        _wputenv_s(L"TEMP", L"C:\\Valid\\Path");
      },
      "C:\\Valid\\Path");

  // All related env vars empty
  EXPECT_TEMP_DIR(
  {
    _wputenv_s(L"TMP", L"");
    _wputenv_s(L"TEMP", L"");
    _wputenv_s(L"USERPROFILE", L"");
  },
    "C:\\Temp");

  // Test evn var / path with 260 chars.
  SmallString<270> Expected{"C:\\Temp\\AB\\123456789"};
  while (Expected.size() < 260)
    Expected.append("\\DirNameWith19Charss");
  ASSERT_EQ(260U, Expected.size());
  EXPECT_TEMP_DIR(_putenv_s("TMP", Expected.c_str()), Expected.c_str());
}
#endif

class FileSystemTest : public testing::Test {
protected:
  /// Unique temporary directory in which all created filesystem entities must
  /// be placed. It is removed at the end of each test (must be empty).
  SmallString<128> TestDirectory;
  SmallString<128> NonExistantFile;

  void SetUp() override {
    ASSERT_NO_ERROR(
        fs::createUniqueDirectory("file-system-test", TestDirectory));
    // We don't care about this specific file.
    errs() << "Test Directory: " << TestDirectory << '\n';
    errs().flush();
    NonExistantFile = TestDirectory;

    // Even though this value is hardcoded, is a 128-bit GUID, so we should be
    // guaranteed that this file will never exist.
    sys::path::append(NonExistantFile, "1B28B495C16344CB9822E588CD4C3EF0");
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
  ASSERT_NO_ERROR(fs::remove(Dir1));
  ASSERT_NO_ERROR(fs::remove(Dir2));
  ASSERT_NO_ERROR(fs::remove(TempPath2));
  ASSERT_NO_ERROR(fs::remove(TempPath));
}

TEST_F(FileSystemTest, RealPath) {
  ASSERT_NO_ERROR(
      fs::create_directories(Twine(TestDirectory) + "/test1/test2/test3"));
  ASSERT_TRUE(fs::exists(Twine(TestDirectory) + "/test1/test2/test3"));

  SmallString<64> RealBase;
  SmallString<64> Expected;
  SmallString<64> Actual;

  // TestDirectory itself might be under a symlink or have been specified with
  // a different case than the existing temp directory.  In such cases real_path
  // on the concatenated path will differ in the TestDirectory portion from
  // how we specified it.  Make sure to compare against the real_path of the
  // TestDirectory, and not just the value of TestDirectory.
  ASSERT_NO_ERROR(fs::real_path(TestDirectory, RealBase));
  path::native(Twine(RealBase) + "/test1/test2", Expected);

  ASSERT_NO_ERROR(fs::real_path(
      Twine(TestDirectory) + "/././test1/../test1/test2/./test3/..", Actual));

  EXPECT_EQ(Expected, Actual);

  SmallString<64> HomeDir;
  bool Result = llvm::sys::path::home_directory(HomeDir);
  if (Result) {
    ASSERT_NO_ERROR(fs::real_path(HomeDir, Expected));
    ASSERT_NO_ERROR(fs::real_path("~", Actual, true));
    EXPECT_EQ(Expected, Actual);
    ASSERT_NO_ERROR(fs::real_path("~/", Actual, true));
    EXPECT_EQ(Expected, Actual);
  }

  ASSERT_NO_ERROR(fs::remove_directories(Twine(TestDirectory) + "/test1"));
}

#ifdef LLVM_ON_UNIX
TEST_F(FileSystemTest, RealPathNoReadPerm) {
  SmallString<64> Expanded;

  ASSERT_NO_ERROR(
    fs::create_directories(Twine(TestDirectory) + "/noreadperm"));
  ASSERT_TRUE(fs::exists(Twine(TestDirectory) + "/noreadperm"));

  fs::setPermissions(Twine(TestDirectory) + "/noreadperm", fs::no_perms);
  fs::setPermissions(Twine(TestDirectory) + "/noreadperm", fs::all_exe);

  ASSERT_NO_ERROR(fs::real_path(Twine(TestDirectory) + "/noreadperm", Expanded,
                                false));

  ASSERT_NO_ERROR(fs::remove_directories(Twine(TestDirectory) + "/noreadperm"));
}
#endif


TEST_F(FileSystemTest, TempFileKeepDiscard) {
  // We can keep then discard.
  auto TempFileOrError = fs::TempFile::create(TestDirectory + "/test-%%%%");
  ASSERT_TRUE((bool)TempFileOrError);
  fs::TempFile File = std::move(*TempFileOrError);
  ASSERT_FALSE((bool)File.keep(TestDirectory + "/keep"));
  ASSERT_FALSE((bool)File.discard());
  ASSERT_TRUE(fs::exists(TestDirectory + "/keep"));
  ASSERT_NO_ERROR(fs::remove(TestDirectory + "/keep"));
}

TEST_F(FileSystemTest, TempFileDiscardDiscard) {
  // We can discard twice.
  auto TempFileOrError = fs::TempFile::create(TestDirectory + "/test-%%%%");
  ASSERT_TRUE((bool)TempFileOrError);
  fs::TempFile File = std::move(*TempFileOrError);
  ASSERT_FALSE((bool)File.discard());
  ASSERT_FALSE((bool)File.discard());
  ASSERT_FALSE(fs::exists(TestDirectory + "/keep"));
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
  FileRemover Cleanup3(TempPath3);

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

#ifdef _WIN32
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

#ifdef _WIN32
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

  // Also verify that paths with Unix separators are handled correctly.
  std::string LongPathWithUnixSeparators(TestDirectory.str());
  // Add at least one subdirectory to TestDirectory, and replace slashes with
  // backslashes
  do {
    LongPathWithUnixSeparators.append("/DirNameWith19Charss");
  } while (LongPathWithUnixSeparators.size() < 260);
  std::replace(LongPathWithUnixSeparators.begin(),
               LongPathWithUnixSeparators.end(),
               '\\', '/');
  ASSERT_NO_ERROR(fs::create_directories(Twine(LongPathWithUnixSeparators)));
  // cleanup
  ASSERT_NO_ERROR(fs::remove_directories(Twine(TestDirectory) +
                                         "/DirNameWith19Charss"));

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
  v_t::const_iterator a0 = find(visited, "a0");
  v_t::const_iterator aa1 = find(visited, "aa1");
  v_t::const_iterator ab1 = find(visited, "ab1");
  v_t::const_iterator dontlookhere = find(visited, "dontlookhere");
  v_t::const_iterator da1 = find(visited, "da1");
  v_t::const_iterator z0 = find(visited, "z0");
  v_t::const_iterator za1 = find(visited, "za1");
  v_t::const_iterator pop = find(visited, "pop");
  v_t::const_iterator p1 = find(visited, "p1");

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

  // Test recursive_directory_iterator level()
  ASSERT_NO_ERROR(
      fs::create_directories(Twine(TestDirectory) + "/reclevel/a/b/c"));
  fs::recursive_directory_iterator I(Twine(TestDirectory) + "/reclevel", ec), E;
  for (int l = 0; I != E; I.increment(ec), ++l) {
    ASSERT_NO_ERROR(ec);
    EXPECT_EQ(I.level(), l);
  }
  EXPECT_EQ(I, E);
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/reclevel/a/b/c"));
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/reclevel/a/b"));
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/reclevel/a"));
  ASSERT_NO_ERROR(fs::remove(Twine(TestDirectory) + "/reclevel"));
}

#ifdef LLVM_ON_UNIX
TEST_F(FileSystemTest, BrokenSymlinkDirectoryIteration) {
  // Create a known hierarchy to recurse over.
  ASSERT_NO_ERROR(fs::create_directories(Twine(TestDirectory) + "/symlink"));
  ASSERT_NO_ERROR(
      fs::create_link("no_such_file", Twine(TestDirectory) + "/symlink/a"));
  ASSERT_NO_ERROR(
      fs::create_directories(Twine(TestDirectory) + "/symlink/b/bb"));
  ASSERT_NO_ERROR(
      fs::create_link("no_such_file", Twine(TestDirectory) + "/symlink/b/ba"));
  ASSERT_NO_ERROR(
      fs::create_link("no_such_file", Twine(TestDirectory) + "/symlink/b/bc"));
  ASSERT_NO_ERROR(
      fs::create_link("no_such_file", Twine(TestDirectory) + "/symlink/c"));
  ASSERT_NO_ERROR(
      fs::create_directories(Twine(TestDirectory) + "/symlink/d/dd/ddd"));
  ASSERT_NO_ERROR(fs::create_link(Twine(TestDirectory) + "/symlink/d/dd",
                                  Twine(TestDirectory) + "/symlink/d/da"));
  ASSERT_NO_ERROR(
      fs::create_link("no_such_file", Twine(TestDirectory) + "/symlink/e"));

  typedef std::vector<std::string> v_t;
  v_t VisitedNonBrokenSymlinks;
  v_t VisitedBrokenSymlinks;
  std::error_code ec;

  // Broken symbol links are expected to throw an error.
  for (fs::directory_iterator i(Twine(TestDirectory) + "/symlink", ec), e;
       i != e; i.increment(ec)) {
    if (ec == std::make_error_code(std::errc::no_such_file_or_directory)) {
      VisitedBrokenSymlinks.push_back(path::filename(i->path()));
      continue;
    }

    ASSERT_NO_ERROR(ec);
    VisitedNonBrokenSymlinks.push_back(path::filename(i->path()));
  }
  llvm::sort(VisitedNonBrokenSymlinks.begin(), VisitedNonBrokenSymlinks.end());
  llvm::sort(VisitedBrokenSymlinks.begin(), VisitedBrokenSymlinks.end());
  v_t ExpectedNonBrokenSymlinks = {"b", "d"};
  ASSERT_EQ(ExpectedNonBrokenSymlinks.size(), VisitedNonBrokenSymlinks.size());
  ASSERT_TRUE(std::equal(VisitedNonBrokenSymlinks.begin(),
                         VisitedNonBrokenSymlinks.end(),
                         ExpectedNonBrokenSymlinks.begin()));
  VisitedNonBrokenSymlinks.clear();

  v_t ExpectedBrokenSymlinks = {"a", "c", "e"};
  ASSERT_EQ(ExpectedBrokenSymlinks.size(), VisitedBrokenSymlinks.size());
  ASSERT_TRUE(std::equal(VisitedBrokenSymlinks.begin(),
                         VisitedBrokenSymlinks.end(),
                         ExpectedBrokenSymlinks.begin()));
  VisitedBrokenSymlinks.clear();

  // Broken symbol links are expected to throw an error.
  for (fs::recursive_directory_iterator i(
      Twine(TestDirectory) + "/symlink", ec), e; i != e; i.increment(ec)) {
    if (ec == std::make_error_code(std::errc::no_such_file_or_directory)) {
      VisitedBrokenSymlinks.push_back(path::filename(i->path()));
      continue;
    }

    ASSERT_NO_ERROR(ec);
    VisitedNonBrokenSymlinks.push_back(path::filename(i->path()));
  }
  llvm::sort(VisitedNonBrokenSymlinks.begin(), VisitedNonBrokenSymlinks.end());
  llvm::sort(VisitedBrokenSymlinks.begin(), VisitedBrokenSymlinks.end());
  ExpectedNonBrokenSymlinks = {"b", "bb", "d", "da", "dd", "ddd", "ddd"};
  ASSERT_EQ(ExpectedNonBrokenSymlinks.size(), VisitedNonBrokenSymlinks.size());
  ASSERT_TRUE(std::equal(VisitedNonBrokenSymlinks.begin(),
                         VisitedNonBrokenSymlinks.end(),
                         ExpectedNonBrokenSymlinks.begin()));
  VisitedNonBrokenSymlinks.clear();

  ExpectedBrokenSymlinks = {"a", "ba", "bc", "c", "e"};
  ASSERT_EQ(ExpectedBrokenSymlinks.size(), VisitedBrokenSymlinks.size());
  ASSERT_TRUE(std::equal(VisitedBrokenSymlinks.begin(),
                         VisitedBrokenSymlinks.end(),
                         ExpectedBrokenSymlinks.begin()));
  VisitedBrokenSymlinks.clear();

  for (fs::recursive_directory_iterator i(
      Twine(TestDirectory) + "/symlink", ec, /*follow_symlinks=*/false), e;
       i != e; i.increment(ec)) {
    if (ec == std::make_error_code(std::errc::no_such_file_or_directory)) {
      VisitedBrokenSymlinks.push_back(path::filename(i->path()));
      continue;
    }

    ASSERT_NO_ERROR(ec);
    VisitedNonBrokenSymlinks.push_back(path::filename(i->path()));
  }
  llvm::sort(VisitedNonBrokenSymlinks.begin(), VisitedNonBrokenSymlinks.end());
  llvm::sort(VisitedBrokenSymlinks.begin(), VisitedBrokenSymlinks.end());
  ExpectedNonBrokenSymlinks = {"a", "b", "ba", "bb", "bc", "c", "d", "da", "dd",
                               "ddd", "e"};
  ASSERT_EQ(ExpectedNonBrokenSymlinks.size(), VisitedNonBrokenSymlinks.size());
  ASSERT_TRUE(std::equal(VisitedNonBrokenSymlinks.begin(),
                         VisitedNonBrokenSymlinks.end(),
                         ExpectedNonBrokenSymlinks.begin()));
  VisitedNonBrokenSymlinks.clear();

  ExpectedBrokenSymlinks = {};
  ASSERT_EQ(ExpectedBrokenSymlinks.size(), VisitedBrokenSymlinks.size());
  ASSERT_TRUE(std::equal(VisitedBrokenSymlinks.begin(),
                         VisitedBrokenSymlinks.end(),
                         ExpectedBrokenSymlinks.begin()));
  VisitedBrokenSymlinks.clear();

  ASSERT_NO_ERROR(fs::remove_directories(Twine(TestDirectory) + "/symlink"));
}
#endif

TEST_F(FileSystemTest, Remove) {
  SmallString<64> BaseDir;
  SmallString<64> Paths[4];
  int fds[4];
  ASSERT_NO_ERROR(fs::createUniqueDirectory("fs_remove", BaseDir));

  ASSERT_NO_ERROR(fs::create_directories(Twine(BaseDir) + "/foo/bar/baz"));
  ASSERT_NO_ERROR(fs::create_directories(Twine(BaseDir) + "/foo/bar/buzz"));
  ASSERT_NO_ERROR(fs::createUniqueFile(
      Twine(BaseDir) + "/foo/bar/baz/%%%%%%.tmp", fds[0], Paths[0]));
  ASSERT_NO_ERROR(fs::createUniqueFile(
      Twine(BaseDir) + "/foo/bar/baz/%%%%%%.tmp", fds[1], Paths[1]));
  ASSERT_NO_ERROR(fs::createUniqueFile(
      Twine(BaseDir) + "/foo/bar/buzz/%%%%%%.tmp", fds[2], Paths[2]));
  ASSERT_NO_ERROR(fs::createUniqueFile(
      Twine(BaseDir) + "/foo/bar/buzz/%%%%%%.tmp", fds[3], Paths[3]));

  for (int fd : fds)
    ::close(fd);

  EXPECT_TRUE(fs::exists(Twine(BaseDir) + "/foo/bar/baz"));
  EXPECT_TRUE(fs::exists(Twine(BaseDir) + "/foo/bar/buzz"));
  EXPECT_TRUE(fs::exists(Paths[0]));
  EXPECT_TRUE(fs::exists(Paths[1]));
  EXPECT_TRUE(fs::exists(Paths[2]));
  EXPECT_TRUE(fs::exists(Paths[3]));

  ASSERT_NO_ERROR(fs::remove_directories("D:/footest"));

  ASSERT_NO_ERROR(fs::remove_directories(BaseDir));
  ASSERT_FALSE(fs::exists(BaseDir));
}

#ifdef _WIN32
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
  ::close(FD);
  ASSERT_NO_ERROR(fs::remove(TempPath));
}

TEST_F(FileSystemTest, MD5) {
  int FD;
  SmallString<64> TempPath;
  ASSERT_NO_ERROR(fs::createTemporaryFile("prefix", "temp", FD, TempPath));
  StringRef Data("abcdefghijklmnopqrstuvwxyz");
  ASSERT_EQ(write(FD, Data.data(), Data.size()), static_cast<ssize_t>(Data.size()));
  lseek(FD, 0, SEEK_SET);
  auto Hash = fs::md5_contents(FD);
  ::close(FD);
  ASSERT_NO_ERROR(Hash.getError());

  EXPECT_STREQ("c3fcd3d76192e4007dfb496cca67e13b", Hash->digest().c_str());
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
  ASSERT_EQ(close(FileDescriptor), 0);

  // Map it back in read-only
  {
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
  ASSERT_NO_ERROR(fs::remove(TempPath));
}

TEST(Support, NormalizePath) {
  using TestTuple = std::tuple<const char *, const char *, const char *>;
  std::vector<TestTuple> Tests;
  Tests.emplace_back("a", "a", "a");
  Tests.emplace_back("a/b", "a\\b", "a/b");
  Tests.emplace_back("a\\b", "a\\b", "a/b");
  Tests.emplace_back("a\\\\b", "a\\\\b", "a\\\\b");
  Tests.emplace_back("\\a", "\\a", "/a");
  Tests.emplace_back("a\\", "a\\", "a/");

  for (auto &T : Tests) {
    SmallString<64> Win(std::get<0>(T));
    SmallString<64> Posix(Win);
    path::native(Win, path::Style::windows);
    path::native(Posix, path::Style::posix);
    EXPECT_EQ(std::get<1>(T), Win);
    EXPECT_EQ(std::get<2>(T), Posix);
  }

#if defined(_WIN32)
  SmallString<64> PathHome;
  path::home_directory(PathHome);

  const char *Path7a = "~/aaa";
  SmallString<64> Path7(Path7a);
  path::native(Path7);
  EXPECT_TRUE(Path7.endswith("\\aaa"));
  EXPECT_TRUE(Path7.startswith(PathHome));
  EXPECT_EQ(Path7.size(), PathHome.size() + strlen(Path7a + 1));

  const char *Path8a = "~";
  SmallString<64> Path8(Path8a);
  path::native(Path8);
  EXPECT_EQ(Path8, PathHome);

  const char *Path9a = "~aaa";
  SmallString<64> Path9(Path9a);
  path::native(Path9);
  EXPECT_EQ(Path9, "~aaa");

  const char *Path10a = "aaa/~/b";
  SmallString<64> Path10(Path10a);
  path::native(Path10);
  EXPECT_EQ(Path10, "aaa\\~\\b");
#endif
}

TEST(Support, RemoveLeadingDotSlash) {
  StringRef Path1("././/foolz/wat");
  StringRef Path2("./////");

  Path1 = path::remove_leading_dotslash(Path1);
  EXPECT_EQ(Path1, "foolz/wat");
  Path2 = path::remove_leading_dotslash(Path2);
  EXPECT_EQ(Path2, "");
}

static std::string remove_dots(StringRef path, bool remove_dot_dot,
                               path::Style style) {
  SmallString<256> buffer(path);
  path::remove_dots(buffer, remove_dot_dot, style);
  return buffer.str();
}

TEST(Support, RemoveDots) {
  EXPECT_EQ("foolz\\wat",
            remove_dots(".\\.\\\\foolz\\wat", false, path::Style::windows));
  EXPECT_EQ("", remove_dots(".\\\\\\\\\\", false, path::Style::windows));

  EXPECT_EQ("a\\..\\b\\c",
            remove_dots(".\\a\\..\\b\\c", false, path::Style::windows));
  EXPECT_EQ("b\\c", remove_dots(".\\a\\..\\b\\c", true, path::Style::windows));
  EXPECT_EQ("c", remove_dots(".\\.\\c", true, path::Style::windows));
  EXPECT_EQ("..\\a\\c",
            remove_dots("..\\a\\b\\..\\c", true, path::Style::windows));
  EXPECT_EQ("..\\..\\a\\c",
            remove_dots("..\\..\\a\\b\\..\\c", true, path::Style::windows));

  SmallString<64> Path1(".\\.\\c");
  EXPECT_TRUE(path::remove_dots(Path1, true, path::Style::windows));
  EXPECT_EQ("c", Path1);

  EXPECT_EQ("foolz/wat",
            remove_dots("././/foolz/wat", false, path::Style::posix));
  EXPECT_EQ("", remove_dots("./////", false, path::Style::posix));

  EXPECT_EQ("a/../b/c", remove_dots("./a/../b/c", false, path::Style::posix));
  EXPECT_EQ("b/c", remove_dots("./a/../b/c", true, path::Style::posix));
  EXPECT_EQ("c", remove_dots("././c", true, path::Style::posix));
  EXPECT_EQ("../a/c", remove_dots("../a/b/../c", true, path::Style::posix));
  EXPECT_EQ("../../a/c",
            remove_dots("../../a/b/../c", true, path::Style::posix));
  EXPECT_EQ("/a/c", remove_dots("/../../a/c", true, path::Style::posix));
  EXPECT_EQ("/a/c",
            remove_dots("/../a/b//../././/c", true, path::Style::posix));

  SmallString<64> Path2("././c");
  EXPECT_TRUE(path::remove_dots(Path2, true, path::Style::posix));
  EXPECT_EQ("c", Path2);
}

TEST(Support, ReplacePathPrefix) {
  SmallString<64> Path1("/foo");
  SmallString<64> Path2("/old/foo");
  SmallString<64> OldPrefix("/old");
  SmallString<64> NewPrefix("/new");
  SmallString<64> NewPrefix2("/longernew");
  SmallString<64> EmptyPrefix("");

  SmallString<64> Path = Path1;
  path::replace_path_prefix(Path, OldPrefix, NewPrefix);
  EXPECT_EQ(Path, "/foo");
  Path = Path2;
  path::replace_path_prefix(Path, OldPrefix, NewPrefix);
  EXPECT_EQ(Path, "/new/foo");
  Path = Path2;
  path::replace_path_prefix(Path, OldPrefix, NewPrefix2);
  EXPECT_EQ(Path, "/longernew/foo");
  Path = Path1;
  path::replace_path_prefix(Path, EmptyPrefix, NewPrefix);
  EXPECT_EQ(Path, "/new/foo");
  Path = Path2;
  path::replace_path_prefix(Path, OldPrefix, EmptyPrefix);
  EXPECT_EQ(Path, "/foo");
}

TEST_F(FileSystemTest, OpenFileForRead) {
  // Create a temp file.
  int FileDescriptor;
  SmallString<64> TempPath;
  ASSERT_NO_ERROR(
      fs::createTemporaryFile("prefix", "temp", FileDescriptor, TempPath));
  FileRemover Cleanup(TempPath);

  // Make sure it exists.
  ASSERT_TRUE(sys::fs::exists(Twine(TempPath)));

  // Open the file for read
  int FileDescriptor2;
  SmallString<64> ResultPath;
  ASSERT_NO_ERROR(fs::openFileForRead(Twine(TempPath), FileDescriptor2,
                                      fs::OF_None, &ResultPath))

  // If we succeeded, check that the paths are the same (modulo case):
  if (!ResultPath.empty()) {
    // The paths returned by createTemporaryFile and getPathFromOpenFD
    // should reference the same file on disk.
    fs::UniqueID D1, D2;
    ASSERT_NO_ERROR(fs::getUniqueID(Twine(TempPath), D1));
    ASSERT_NO_ERROR(fs::getUniqueID(Twine(ResultPath), D2));
    ASSERT_EQ(D1, D2);
  }
  ::close(FileDescriptor);
  ::close(FileDescriptor2);

#ifdef _WIN32
  // Since Windows Vista, file access time is not updated by default.
  // This is instead updated manually by openFileForRead.
  // https://blogs.technet.microsoft.com/filecab/2006/11/07/disabling-last-access-time-in-windows-vista-to-improve-ntfs-performance/
  // This part of the unit test is Windows specific as the updating of
  // access times can be disabled on Linux using /etc/fstab.

  // Set access time to UNIX epoch.
  ASSERT_NO_ERROR(sys::fs::openFileForWrite(Twine(TempPath), FileDescriptor,
                                            fs::CD_OpenExisting));
  TimePoint<> Epoch(std::chrono::milliseconds(0));
  ASSERT_NO_ERROR(fs::setLastModificationAndAccessTime(FileDescriptor, Epoch));
  ::close(FileDescriptor);

  // Open the file and ensure access time is updated, when forced.
  ASSERT_NO_ERROR(fs::openFileForRead(Twine(TempPath), FileDescriptor,
                                      fs::OF_UpdateAtime, &ResultPath));

  sys::fs::file_status Status;
  ASSERT_NO_ERROR(sys::fs::status(FileDescriptor, Status));
  auto FileAccessTime = Status.getLastAccessedTime();

  ASSERT_NE(Epoch, FileAccessTime);
  ::close(FileDescriptor);

  // Ideally this test would include a case when ATime is not forced to update,
  // however the expected behaviour will differ depending on the configuration
  // of the Windows file system.
#endif
}

static void createFileWithData(const Twine &Path, bool ShouldExistBefore,
                               fs::CreationDisposition Disp, StringRef Data) {
  int FD;
  ASSERT_EQ(ShouldExistBefore, fs::exists(Path));
  ASSERT_NO_ERROR(fs::openFileForWrite(Path, FD, Disp));
  FileDescriptorCloser Closer(FD);
  ASSERT_TRUE(fs::exists(Path));

  ASSERT_EQ(Data.size(), (size_t)write(FD, Data.data(), Data.size()));
}

static void verifyFileContents(const Twine &Path, StringRef Contents) {
  auto Buffer = MemoryBuffer::getFile(Path);
  ASSERT_TRUE((bool)Buffer);
  StringRef Data = Buffer.get()->getBuffer();
  ASSERT_EQ(Data, Contents);
}

TEST_F(FileSystemTest, CreateNew) {
  int FD;
  Optional<FileDescriptorCloser> Closer;

  // Succeeds if the file does not exist.
  ASSERT_FALSE(fs::exists(NonExistantFile));
  ASSERT_NO_ERROR(fs::openFileForWrite(NonExistantFile, FD, fs::CD_CreateNew));
  ASSERT_TRUE(fs::exists(NonExistantFile));

  FileRemover Cleanup(NonExistantFile);
  Closer.emplace(FD);

  // And creates a file of size 0.
  sys::fs::file_status Status;
  ASSERT_NO_ERROR(sys::fs::status(FD, Status));
  EXPECT_EQ(0ULL, Status.getSize());

  // Close this first, before trying to re-open the file.
  Closer.reset();

  // But fails if the file does exist.
  ASSERT_ERROR(fs::openFileForWrite(NonExistantFile, FD, fs::CD_CreateNew));
}

TEST_F(FileSystemTest, CreateAlways) {
  int FD;
  Optional<FileDescriptorCloser> Closer;

  // Succeeds if the file does not exist.
  ASSERT_FALSE(fs::exists(NonExistantFile));
  ASSERT_NO_ERROR(
      fs::openFileForWrite(NonExistantFile, FD, fs::CD_CreateAlways));

  Closer.emplace(FD);

  ASSERT_TRUE(fs::exists(NonExistantFile));

  FileRemover Cleanup(NonExistantFile);

  // And creates a file of size 0.
  uint64_t FileSize;
  ASSERT_NO_ERROR(sys::fs::file_size(NonExistantFile, FileSize));
  ASSERT_EQ(0ULL, FileSize);

  // If we write some data to it re-create it with CreateAlways, it succeeds and
  // truncates to 0 bytes.
  ASSERT_EQ(4, write(FD, "Test", 4));

  Closer.reset();

  ASSERT_NO_ERROR(sys::fs::file_size(NonExistantFile, FileSize));
  ASSERT_EQ(4ULL, FileSize);

  ASSERT_NO_ERROR(
      fs::openFileForWrite(NonExistantFile, FD, fs::CD_CreateAlways));
  Closer.emplace(FD);
  ASSERT_NO_ERROR(sys::fs::file_size(NonExistantFile, FileSize));
  ASSERT_EQ(0ULL, FileSize);
}

TEST_F(FileSystemTest, OpenExisting) {
  int FD;

  // Fails if the file does not exist.
  ASSERT_FALSE(fs::exists(NonExistantFile));
  ASSERT_ERROR(fs::openFileForWrite(NonExistantFile, FD, fs::CD_OpenExisting));
  ASSERT_FALSE(fs::exists(NonExistantFile));

  // Make a dummy file now so that we can try again when the file does exist.
  createFileWithData(NonExistantFile, false, fs::CD_CreateNew, "Fizz");
  FileRemover Cleanup(NonExistantFile);
  uint64_t FileSize;
  ASSERT_NO_ERROR(sys::fs::file_size(NonExistantFile, FileSize));
  ASSERT_EQ(4ULL, FileSize);

  // If we re-create it with different data, it overwrites rather than
  // appending.
  createFileWithData(NonExistantFile, true, fs::CD_OpenExisting, "Buzz");
  verifyFileContents(NonExistantFile, "Buzz");
}

TEST_F(FileSystemTest, OpenAlways) {
  // Succeeds if the file does not exist.
  createFileWithData(NonExistantFile, false, fs::CD_OpenAlways, "Fizz");
  FileRemover Cleanup(NonExistantFile);
  uint64_t FileSize;
  ASSERT_NO_ERROR(sys::fs::file_size(NonExistantFile, FileSize));
  ASSERT_EQ(4ULL, FileSize);

  // Now re-open it and write again, verifying the contents get over-written.
  createFileWithData(NonExistantFile, true, fs::CD_OpenAlways, "Bu");
  verifyFileContents(NonExistantFile, "Buzz");
}

TEST_F(FileSystemTest, AppendSetsCorrectFileOffset) {
  fs::CreationDisposition Disps[] = {fs::CD_CreateAlways, fs::CD_OpenAlways,
                                     fs::CD_OpenExisting};

  // Write some data and re-open it with every possible disposition (this is a
  // hack that shouldn't work, but is left for compatibility.  F_Append
  // overrides
  // the specified disposition.
  for (fs::CreationDisposition Disp : Disps) {
    int FD;
    Optional<FileDescriptorCloser> Closer;

    createFileWithData(NonExistantFile, false, fs::CD_CreateNew, "Fizz");

    FileRemover Cleanup(NonExistantFile);

    uint64_t FileSize;
    ASSERT_NO_ERROR(sys::fs::file_size(NonExistantFile, FileSize));
    ASSERT_EQ(4ULL, FileSize);
    ASSERT_NO_ERROR(
        fs::openFileForWrite(NonExistantFile, FD, Disp, fs::OF_Append));
    Closer.emplace(FD);
    ASSERT_NO_ERROR(sys::fs::file_size(NonExistantFile, FileSize));
    ASSERT_EQ(4ULL, FileSize);

    ASSERT_EQ(4, write(FD, "Buzz", 4));
    Closer.reset();

    verifyFileContents(NonExistantFile, "FizzBuzz");
  }
}

static void verifyRead(int FD, StringRef Data, bool ShouldSucceed) {
  std::vector<char> Buffer;
  Buffer.resize(Data.size());
  int Result = ::read(FD, Buffer.data(), Buffer.size());
  if (ShouldSucceed) {
    ASSERT_EQ((size_t)Result, Data.size());
    ASSERT_EQ(Data, StringRef(Buffer.data(), Buffer.size()));
  } else {
    ASSERT_EQ(-1, Result);
    ASSERT_EQ(EBADF, errno);
  }
}

static void verifyWrite(int FD, StringRef Data, bool ShouldSucceed) {
  int Result = ::write(FD, Data.data(), Data.size());
  if (ShouldSucceed)
    ASSERT_EQ((size_t)Result, Data.size());
  else {
    ASSERT_EQ(-1, Result);
    ASSERT_EQ(EBADF, errno);
  }
}

TEST_F(FileSystemTest, ReadOnlyFileCantWrite) {
  createFileWithData(NonExistantFile, false, fs::CD_CreateNew, "Fizz");
  FileRemover Cleanup(NonExistantFile);

  int FD;
  ASSERT_NO_ERROR(fs::openFileForRead(NonExistantFile, FD));
  FileDescriptorCloser Closer(FD);

  verifyWrite(FD, "Buzz", false);
  verifyRead(FD, "Fizz", true);
}

TEST_F(FileSystemTest, WriteOnlyFileCantRead) {
  createFileWithData(NonExistantFile, false, fs::CD_CreateNew, "Fizz");
  FileRemover Cleanup(NonExistantFile);

  int FD;
  ASSERT_NO_ERROR(
      fs::openFileForWrite(NonExistantFile, FD, fs::CD_OpenExisting));
  FileDescriptorCloser Closer(FD);
  verifyRead(FD, "Fizz", false);
  verifyWrite(FD, "Buzz", true);
}

TEST_F(FileSystemTest, ReadWriteFileCanReadOrWrite) {
  createFileWithData(NonExistantFile, false, fs::CD_CreateNew, "Fizz");
  FileRemover Cleanup(NonExistantFile);

  int FD;
  ASSERT_NO_ERROR(fs::openFileForReadWrite(NonExistantFile, FD,
                                           fs::CD_OpenExisting, fs::OF_None));
  FileDescriptorCloser Closer(FD);
  verifyRead(FD, "Fizz", true);
  verifyWrite(FD, "Buzz", true);
}

TEST_F(FileSystemTest, set_current_path) {
  SmallString<128> path;

  ASSERT_NO_ERROR(fs::current_path(path));
  ASSERT_NE(TestDirectory, path);

  struct RestorePath {
    SmallString<128> path;
    RestorePath(const SmallString<128> &path) : path(path) {}
    ~RestorePath() { fs::set_current_path(path); }
  } restore_path(path);

  ASSERT_NO_ERROR(fs::set_current_path(TestDirectory));

  ASSERT_NO_ERROR(fs::current_path(path));

  fs::UniqueID D1, D2;
  ASSERT_NO_ERROR(fs::getUniqueID(TestDirectory, D1));
  ASSERT_NO_ERROR(fs::getUniqueID(path, D2));
  ASSERT_EQ(D1, D2) << "D1: " << TestDirectory << "\nD2: " << path;
}

TEST_F(FileSystemTest, permissions) {
  int FD;
  SmallString<64> TempPath;
  ASSERT_NO_ERROR(fs::createTemporaryFile("prefix", "temp", FD, TempPath));
  FileRemover Cleanup(TempPath);

  // Make sure it exists.
  ASSERT_TRUE(fs::exists(Twine(TempPath)));

  auto CheckPermissions = [&](fs::perms Expected) {
    ErrorOr<fs::perms> Actual = fs::getPermissions(TempPath);
    return Actual && *Actual == Expected;
  };

  std::error_code NoError;
  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_all), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_all));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_read | fs::all_exe), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_read | fs::all_exe));

#if defined(_WIN32)
  fs::perms ReadOnly = fs::all_read | fs::all_exe;
  EXPECT_EQ(fs::setPermissions(TempPath, fs::no_perms), NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::owner_read), NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::owner_write), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_all));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::owner_exe), NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::owner_all), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_all));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::group_read), NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::group_write), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_all));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::group_exe), NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::group_all), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_all));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::others_read), NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::others_write), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_all));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::others_exe), NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::others_all), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_all));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_read), NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_write), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_all));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_exe), NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::set_uid_on_exe), NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::set_gid_on_exe), NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::sticky_bit), NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::set_uid_on_exe |
                                             fs::set_gid_on_exe |
                                             fs::sticky_bit),
            NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, ReadOnly | fs::set_uid_on_exe |
                                             fs::set_gid_on_exe |
                                             fs::sticky_bit),
            NoError);
  EXPECT_TRUE(CheckPermissions(ReadOnly));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_perms), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_all));
#else
  EXPECT_EQ(fs::setPermissions(TempPath, fs::no_perms), NoError);
  EXPECT_TRUE(CheckPermissions(fs::no_perms));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::owner_read), NoError);
  EXPECT_TRUE(CheckPermissions(fs::owner_read));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::owner_write), NoError);
  EXPECT_TRUE(CheckPermissions(fs::owner_write));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::owner_exe), NoError);
  EXPECT_TRUE(CheckPermissions(fs::owner_exe));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::owner_all), NoError);
  EXPECT_TRUE(CheckPermissions(fs::owner_all));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::group_read), NoError);
  EXPECT_TRUE(CheckPermissions(fs::group_read));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::group_write), NoError);
  EXPECT_TRUE(CheckPermissions(fs::group_write));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::group_exe), NoError);
  EXPECT_TRUE(CheckPermissions(fs::group_exe));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::group_all), NoError);
  EXPECT_TRUE(CheckPermissions(fs::group_all));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::others_read), NoError);
  EXPECT_TRUE(CheckPermissions(fs::others_read));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::others_write), NoError);
  EXPECT_TRUE(CheckPermissions(fs::others_write));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::others_exe), NoError);
  EXPECT_TRUE(CheckPermissions(fs::others_exe));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::others_all), NoError);
  EXPECT_TRUE(CheckPermissions(fs::others_all));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_read), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_read));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_write), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_write));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_exe), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_exe));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::set_uid_on_exe), NoError);
  EXPECT_TRUE(CheckPermissions(fs::set_uid_on_exe));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::set_gid_on_exe), NoError);
  EXPECT_TRUE(CheckPermissions(fs::set_gid_on_exe));

  // Modern BSDs require root to set the sticky bit on files.
#if !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__)
  EXPECT_EQ(fs::setPermissions(TempPath, fs::sticky_bit), NoError);
  EXPECT_TRUE(CheckPermissions(fs::sticky_bit));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::set_uid_on_exe |
                                             fs::set_gid_on_exe |
                                             fs::sticky_bit),
            NoError);
  EXPECT_TRUE(CheckPermissions(fs::set_uid_on_exe | fs::set_gid_on_exe |
                               fs::sticky_bit));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_read | fs::set_uid_on_exe |
                                             fs::set_gid_on_exe |
                                             fs::sticky_bit),
            NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_read | fs::set_uid_on_exe |
                               fs::set_gid_on_exe | fs::sticky_bit));

  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_perms), NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_perms));
#endif // !FreeBSD && !NetBSD && !OpenBSD

  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_perms & ~fs::sticky_bit),
                               NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_perms & ~fs::sticky_bit));
#endif
}

} // anonymous namespace
