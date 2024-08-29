//===- llvm/unittest/Support/Path.cpp - Path tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Path.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Duration.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#ifdef _WIN32
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/Windows/WindowsSupport.h"
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

void checkSeparators(StringRef Path) {
#ifdef _WIN32
  char UndesiredSeparator = sys::path::get_separator()[0] == '/' ? '\\' : '/';
  ASSERT_EQ(Path.find(UndesiredSeparator), StringRef::npos);
#endif
}

struct FileDescriptorCloser {
  explicit FileDescriptorCloser(int FD) : FD(FD) {}
  ~FileDescriptorCloser() { ::close(FD); }
  int FD;
};

TEST(is_style_Style, Works) {
  using namespace llvm::sys::path;
  // Check platform-independent results.
  EXPECT_TRUE(is_style_posix(Style::posix));
  EXPECT_TRUE(is_style_windows(Style::windows));
  EXPECT_TRUE(is_style_windows(Style::windows_slash));
  EXPECT_FALSE(is_style_posix(Style::windows));
  EXPECT_FALSE(is_style_posix(Style::windows_slash));
  EXPECT_FALSE(is_style_windows(Style::posix));

  // Check platform-dependent results.
#if defined(_WIN32)
  EXPECT_FALSE(is_style_posix(Style::native));
  EXPECT_TRUE(is_style_windows(Style::native));
#else
  EXPECT_TRUE(is_style_posix(Style::native));
  EXPECT_FALSE(is_style_windows(Style::native));
#endif
}

TEST(is_separator, Works) {
  EXPECT_TRUE(path::is_separator('/'));
  EXPECT_FALSE(path::is_separator('\0'));
  EXPECT_FALSE(path::is_separator('-'));
  EXPECT_FALSE(path::is_separator(' '));

  EXPECT_TRUE(path::is_separator('\\', path::Style::windows));
  EXPECT_TRUE(path::is_separator('\\', path::Style::windows_slash));
  EXPECT_FALSE(path::is_separator('\\', path::Style::posix));

  EXPECT_EQ(path::is_style_windows(path::Style::native),
            path::is_separator('\\'));
}

TEST(get_separator, Works) {
  EXPECT_EQ(path::get_separator(path::Style::posix), "/");
  EXPECT_EQ(path::get_separator(path::Style::windows_backslash), "\\");
  EXPECT_EQ(path::get_separator(path::Style::windows_slash), "/");
}

TEST(is_absolute_gnu, Works) {
  // Test tuple <Path, ExpectedPosixValue, ExpectedWindowsValue>.
  const std::tuple<StringRef, bool, bool> Paths[] = {
      std::make_tuple("", false, false),
      std::make_tuple("/", true, true),
      std::make_tuple("/foo", true, true),
      std::make_tuple("\\", false, true),
      std::make_tuple("\\foo", false, true),
      std::make_tuple("foo", false, false),
      std::make_tuple("c", false, false),
      std::make_tuple("c:", false, true),
      std::make_tuple("c:\\", false, true),
      std::make_tuple("!:", false, true),
      std::make_tuple("xx:", false, false),
      std::make_tuple("c:abc\\", false, true),
      std::make_tuple(":", false, false)};

  for (const auto &Path : Paths) {
    EXPECT_EQ(path::is_absolute_gnu(std::get<0>(Path), path::Style::posix),
              std::get<1>(Path));
    EXPECT_EQ(path::is_absolute_gnu(std::get<0>(Path), path::Style::windows),
              std::get<2>(Path));

    constexpr int Native = is_style_posix(path::Style::native) ? 1 : 2;
    EXPECT_EQ(path::is_absolute_gnu(std::get<0>(Path), path::Style::native),
              std::get<Native>(Path));
  }
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
  paths.push_back(":");

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
    (void)path::is_absolute_gnu(*i);
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

  {
    SmallString<32> Relative("foo.cpp");
    sys::fs::make_absolute("/root", Relative);
    Relative[5] = '/'; // Fix up windows paths.
    ASSERT_EQ("/root/foo.cpp", Relative);
  }

  {
    SmallString<32> Relative("foo.cpp");
    sys::fs::make_absolute("//root", Relative);
    Relative[6] = '/'; // Fix up windows paths.
    ASSERT_EQ("//root/foo.cpp", Relative);
  }
}

TEST(Support, PathRoot) {
  ASSERT_EQ(path::root_name("//net/hello", path::Style::posix).str(), "//net");
  ASSERT_EQ(path::root_name("c:/hello", path::Style::posix).str(), "");
  ASSERT_EQ(path::root_name("c:/hello", path::Style::windows).str(), "c:");
  ASSERT_EQ(path::root_name("/hello", path::Style::posix).str(), "");

  ASSERT_EQ(path::root_directory("/goo/hello", path::Style::posix).str(), "/");
  ASSERT_EQ(path::root_directory("c:/hello", path::Style::windows).str(), "/");
  ASSERT_EQ(path::root_directory("d/file.txt", path::Style::posix).str(), "");
  ASSERT_EQ(path::root_directory("d/file.txt", path::Style::windows).str(), "");

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

  for (StringRef p : paths) {
    ASSERT_EQ(
      path::root_name(p, path::Style::posix).str() + path::root_directory(p, path::Style::posix).str(),
      path::root_path(p, path::Style::posix).str());

    ASSERT_EQ(
      path::root_name(p, path::Style::windows).str() + path::root_directory(p, path::Style::windows).str(),
      path::root_path(p, path::Style::windows).str());
  }
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
  EXPECT_THAT(GetComponents("c:\\c\\e\\foo.txt", path::Style::windows_slash),
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

#ifdef _WIN32
std::string getEnvWin(const wchar_t *Var) {
  std::string expected;
  if (wchar_t const *path = ::_wgetenv(Var)) {
    auto pathLen = ::wcslen(path);
    ArrayRef<char> ref{reinterpret_cast<char const *>(path),
                       pathLen * sizeof(wchar_t)};
    convertUTF16ToUTF8String(ref, expected);
    SmallString<32> Buf(expected);
    path::make_preferred(Buf);
    expected.assign(Buf.begin(), Buf.end());
  }
  return expected;
}
#else
// RAII helper to set and restore an environment variable.
class WithEnv {
  const char *Var;
  std::optional<std::string> OriginalValue;

public:
  WithEnv(const char *Var, const char *Value) : Var(Var) {
    if (const char *V = ::getenv(Var))
      OriginalValue.emplace(V);
    if (Value)
      ::setenv(Var, Value, 1);
    else
      ::unsetenv(Var);
  }
  ~WithEnv() {
    if (OriginalValue)
      ::setenv(Var, OriginalValue->c_str(), 1);
    else
      ::unsetenv(Var);
  }
};
#endif

TEST(Support, HomeDirectory) {
  std::string expected;
#ifdef _WIN32
  expected = getEnvWin(L"USERPROFILE");
#else
  if (char const *path = ::getenv("HOME"))
    expected = path;
#endif
  // Do not try to test it if we don't know what to expect.
  // On Windows we use something better than env vars.
  if (expected.empty())
    GTEST_SKIP();
  SmallString<128> HomeDir;
  auto status = path::home_directory(HomeDir);
  EXPECT_TRUE(status);
  EXPECT_EQ(expected, HomeDir);
}

// Apple has their own solution for this.
#if defined(LLVM_ON_UNIX) && !defined(__APPLE__)
TEST(Support, HomeDirectoryWithNoEnv) {
  WithEnv Env("HOME", nullptr);

  // Don't run the test if we have nothing to compare against.
  struct passwd *pw = getpwuid(getuid());
  if (!pw || !pw->pw_dir)
    GTEST_SKIP();
  std::string PwDir = pw->pw_dir;

  SmallString<128> HomeDir;
  EXPECT_TRUE(path::home_directory(HomeDir));
  EXPECT_EQ(PwDir, HomeDir);
}

TEST(Support, ConfigDirectoryWithEnv) {
  WithEnv Env("XDG_CONFIG_HOME", "/xdg/config");

  SmallString<128> ConfigDir;
  EXPECT_TRUE(path::user_config_directory(ConfigDir));
  EXPECT_EQ("/xdg/config", ConfigDir);
}

TEST(Support, ConfigDirectoryNoEnv) {
  WithEnv Env("XDG_CONFIG_HOME", nullptr);

  SmallString<128> Fallback;
  ASSERT_TRUE(path::home_directory(Fallback));
  path::append(Fallback, ".config");

  SmallString<128> CacheDir;
  EXPECT_TRUE(path::user_config_directory(CacheDir));
  EXPECT_EQ(Fallback, CacheDir);
}

TEST(Support, CacheDirectoryWithEnv) {
  WithEnv Env("XDG_CACHE_HOME", "/xdg/cache");

  SmallString<128> CacheDir;
  EXPECT_TRUE(path::cache_directory(CacheDir));
  EXPECT_EQ("/xdg/cache", CacheDir);
}

TEST(Support, CacheDirectoryNoEnv) {
  WithEnv Env("XDG_CACHE_HOME", nullptr);

  SmallString<128> Fallback;
  ASSERT_TRUE(path::home_directory(Fallback));
  path::append(Fallback, ".cache");

  SmallString<128> CacheDir;
  EXPECT_TRUE(path::cache_directory(CacheDir));
  EXPECT_EQ(Fallback, CacheDir);
}
#endif

#ifdef __APPLE__
TEST(Support, ConfigDirectory) {
  SmallString<128> Fallback;
  ASSERT_TRUE(path::home_directory(Fallback));
  path::append(Fallback, "Library/Preferences");

  SmallString<128> ConfigDir;
  EXPECT_TRUE(path::user_config_directory(ConfigDir));
  EXPECT_EQ(Fallback, ConfigDir);
}
#endif

#ifdef _WIN32
TEST(Support, ConfigDirectory) {
  std::string Expected = getEnvWin(L"LOCALAPPDATA");
  // Do not try to test it if we don't know what to expect.
  if (Expected.empty())
    GTEST_SKIP();
  SmallString<128> CacheDir;
  EXPECT_TRUE(path::user_config_directory(CacheDir));
  EXPECT_EQ(Expected, CacheDir);
}

TEST(Support, CacheDirectory) {
  std::string Expected = getEnvWin(L"LOCALAPPDATA");
  // Do not try to test it if we don't know what to expect.
  if (Expected.empty())
    GTEST_SKIP();
  SmallString<128> CacheDir;
  EXPECT_TRUE(path::cache_directory(CacheDir));
  EXPECT_EQ(Expected, CacheDir);
}
#endif

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
  bool Forward = path::get_separator()[0] == '/';
  while ((Pos = Path.find('\\', Pos)) != std::string::npos) {
    if (Forward) {
      Path.replace(Pos, 1, "/");
      Pos += 1;
    } else {
      Path.replace(Pos, 1, "\\\\");
      Pos += 2;
    }
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
  EXPECT_TEMP_DIR(_wputenv_s(L"TMP", L"C:/Unix/Path/Separators"),
                  "C:\\Unix\\Path\\Separators");
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

#ifndef _WIN32
  // Two paths representing the same file on disk should still provide the
  // same unique id.  We can test this by making a hard link.
  // FIXME: Our implementation of getUniqueID on Windows doesn't consider hard
  // links to be the same file.
  ASSERT_NO_ERROR(fs::create_link(Twine(TempPath), Twine(TempPath2)));
  fs::UniqueID D2;
  ASSERT_NO_ERROR(fs::getUniqueID(Twine(TempPath2), D2));
  ASSERT_EQ(D2, F1);
#endif

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
  checkSeparators(RealBase);
  path::native(Twine(RealBase) + "/test1/test2", Expected);

  ASSERT_NO_ERROR(fs::real_path(
      Twine(TestDirectory) + "/././test1/../test1/test2/./test3/..", Actual));
  checkSeparators(Actual);

  EXPECT_EQ(Expected, Actual);

  SmallString<64> HomeDir;

  // This can fail if $HOME is not set and getpwuid fails.
  bool Result = llvm::sys::path::home_directory(HomeDir);
  if (Result) {
    checkSeparators(HomeDir);
    ASSERT_NO_ERROR(fs::real_path(HomeDir, Expected));
    checkSeparators(Expected);
    ASSERT_NO_ERROR(fs::real_path("~", Actual, true));
    EXPECT_EQ(Expected, Actual);
    ASSERT_NO_ERROR(fs::real_path("~/", Actual, true));
    EXPECT_EQ(Expected, Actual);
  }

  ASSERT_NO_ERROR(fs::remove_directories(Twine(TestDirectory) + "/test1"));
}

TEST_F(FileSystemTest, ExpandTilde) {
  SmallString<64> Expected;
  SmallString<64> Actual;
  SmallString<64> HomeDir;

  // This can fail if $HOME is not set and getpwuid fails.
  bool Result = llvm::sys::path::home_directory(HomeDir);
  if (Result) {
    fs::expand_tilde(HomeDir, Expected);

    fs::expand_tilde("~", Actual);
    EXPECT_EQ(Expected, Actual);

#ifdef _WIN32
    Expected += "\\foo";
    fs::expand_tilde("~\\foo", Actual);
#else
    Expected += "/foo";
    fs::expand_tilde("~/foo", Actual);
#endif

    EXPECT_EQ(Expected, Actual);
  }
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
TEST_F(FileSystemTest, RemoveDirectoriesNoExePerm) {
  SmallString<64> Expanded;

  ASSERT_NO_ERROR(
      fs::create_directories(Twine(TestDirectory) + "/noexeperm/foo"));
  ASSERT_TRUE(fs::exists(Twine(TestDirectory) + "/noexeperm/foo"));

  fs::setPermissions(Twine(TestDirectory) + "/noexeperm",
                     fs::all_read | fs::all_write);

  ASSERT_NO_ERROR(fs::remove_directories(Twine(TestDirectory) + "/noexeperm",
                                         /*IgnoreErrors=*/true));

  // It's expected that the directory exists, but some environments appear to
  // allow the removal despite missing the 'x' permission, so be flexible.
  if (fs::exists(Twine(TestDirectory) + "/noexeperm")) {
    fs::setPermissions(Twine(TestDirectory) + "/noexeperm", fs::all_perms);
    ASSERT_NO_ERROR(fs::remove_directories(Twine(TestDirectory) + "/noexeperm",
                                           /*IgnoreErrors=*/false));
  }
}
#endif


TEST_F(FileSystemTest, TempFileKeepDiscard) {
  // We can keep then discard.
  auto TempFileOrError = fs::TempFile::create(TestDirectory + "/test-%%%%");
  ASSERT_TRUE((bool)TempFileOrError);
  fs::TempFile File = std::move(*TempFileOrError);
  ASSERT_EQ(-1, TempFileOrError->FD);
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
  ASSERT_EQ(-1, TempFileOrError->FD);
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
  ASSERT_TRUE(TempPath2.ends_with(".temp"));
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
  ASSERT_FALSE(TempPath3.ends_with("."));
  FileRemover Cleanup3(TempPath3);

  // Create a hard link to Temp1.
  ASSERT_NO_ERROR(fs::create_link(Twine(TempPath), Twine(TempPath2)));
#ifndef _WIN32
  // FIXME: Our implementation of equivalent() on Windows doesn't consider hard
  // links to be the same file.
  bool equal;
  ASSERT_NO_ERROR(fs::equivalent(Twine(TempPath), Twine(TempPath2), equal));
  EXPECT_TRUE(equal);
  ASSERT_NO_ERROR(fs::status(Twine(TempPath), A));
  ASSERT_NO_ERROR(fs::status(Twine(TempPath2), B));
  EXPECT_TRUE(fs::equivalent(A, B));
#endif

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

TEST_F(FileSystemTest, TempFileCollisions) {
  SmallString<128> TestDirectory;
  ASSERT_NO_ERROR(
      fs::createUniqueDirectory("CreateUniqueFileTest", TestDirectory));
  FileRemover Cleanup(TestDirectory);
  SmallString<128> Model = TestDirectory;
  path::append(Model, "%.tmp");
  SmallString<128> Path;
  std::vector<fs::TempFile> TempFiles;

  auto TryCreateTempFile = [&]() {
    Expected<fs::TempFile> T = fs::TempFile::create(Model);
    if (T) {
      TempFiles.push_back(std::move(*T));
      return true;
    } else {
      logAllUnhandledErrors(T.takeError(), errs(),
                            "Failed to create temporary file: ");
      return false;
    }
  };

  // Our single-character template allows for 16 unique names. Check that
  // calling TryCreateTempFile repeatedly results in 16 successes.
  // Because the test depends on random numbers, it could theoretically fail.
  // However, the probability of this happening is tiny: with 32 calls, each
  // of which will retry up to 128 times, to not get a given digit we would
  // have to fail at least 15 + 17 * 128 = 2191 attempts. The probability of
  // 2191 attempts not producing a given hexadecimal digit is
  // (1 - 1/16) ** 2191 or 3.88e-62.
  int Successes = 0;
  for (int i = 0; i < 32; ++i)
    if (TryCreateTempFile()) ++Successes;
  EXPECT_EQ(Successes, 16);

  for (fs::TempFile &T : TempFiles)
    cantFail(T.discard());
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
    visited.push_back(std::string(path::filename(i->path())));
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

TEST_F(FileSystemTest, DirectoryNotExecutable) {
  ASSERT_EQ(fs::access(TestDirectory, sys::fs::AccessMode::Execute),
            errc::permission_denied);
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
  using testing::UnorderedElementsAre;
  using testing::UnorderedElementsAreArray;

  // Broken symbol links are expected to throw an error.
  for (fs::directory_iterator i(Twine(TestDirectory) + "/symlink", ec), e;
       i != e; i.increment(ec)) {
    ASSERT_NO_ERROR(ec);
    if (i->status().getError() ==
        std::make_error_code(std::errc::no_such_file_or_directory)) {
      VisitedBrokenSymlinks.push_back(std::string(path::filename(i->path())));
      continue;
    }
    VisitedNonBrokenSymlinks.push_back(std::string(path::filename(i->path())));
  }
  EXPECT_THAT(VisitedNonBrokenSymlinks, UnorderedElementsAre("b", "d"));
  VisitedNonBrokenSymlinks.clear();

  EXPECT_THAT(VisitedBrokenSymlinks, UnorderedElementsAre("a", "c", "e"));
  VisitedBrokenSymlinks.clear();

  // Broken symbol links are expected to throw an error.
  for (fs::recursive_directory_iterator i(
      Twine(TestDirectory) + "/symlink", ec), e; i != e; i.increment(ec)) {
    ASSERT_NO_ERROR(ec);
    if (i->status().getError() ==
        std::make_error_code(std::errc::no_such_file_or_directory)) {
      VisitedBrokenSymlinks.push_back(std::string(path::filename(i->path())));
      continue;
    }
    VisitedNonBrokenSymlinks.push_back(std::string(path::filename(i->path())));
  }
  EXPECT_THAT(VisitedNonBrokenSymlinks,
              UnorderedElementsAre("b", "bb", "d", "da", "dd", "ddd", "ddd"));
  VisitedNonBrokenSymlinks.clear();

  EXPECT_THAT(VisitedBrokenSymlinks,
              UnorderedElementsAre("a", "ba", "bc", "c", "e"));
  VisitedBrokenSymlinks.clear();

  for (fs::recursive_directory_iterator i(
      Twine(TestDirectory) + "/symlink", ec, /*follow_symlinks=*/false), e;
       i != e; i.increment(ec)) {
    ASSERT_NO_ERROR(ec);
    if (i->status().getError() ==
        std::make_error_code(std::errc::no_such_file_or_directory)) {
      VisitedBrokenSymlinks.push_back(std::string(path::filename(i->path())));
      continue;
    }
    VisitedNonBrokenSymlinks.push_back(std::string(path::filename(i->path())));
  }
  EXPECT_THAT(VisitedNonBrokenSymlinks,
              UnorderedElementsAreArray({"a", "b", "ba", "bb", "bc", "c", "d",
                                         "da", "dd", "ddd", "e"}));
  VisitedNonBrokenSymlinks.clear();

  EXPECT_THAT(VisitedBrokenSymlinks, UnorderedElementsAre());
  VisitedBrokenSymlinks.clear();

  ASSERT_NO_ERROR(fs::remove_directories(Twine(TestDirectory) + "/symlink"));
}
#endif

#ifdef _WIN32
TEST_F(FileSystemTest, UTF8ToUTF16DirectoryIteration) {
  // The Windows filesystem support uses UTF-16 and converts paths from the
  // input UTF-8. The UTF-16 equivalent of the input path can be shorter in
  // length.

  // This test relies on TestDirectory not being so long such that MAX_PATH
  // would be exceeded (see widenPath). If that were the case, the UTF-16
  // path is likely to be longer than the input.
  const char *Pi = "\xcf\x80"; // UTF-8 lower case pi.
  std::string RootDir = (TestDirectory + "/" + Pi).str();

  // Create test directories.
  ASSERT_NO_ERROR(fs::create_directories(Twine(RootDir) + "/a"));
  ASSERT_NO_ERROR(fs::create_directories(Twine(RootDir) + "/b"));

  std::error_code EC;
  unsigned Count = 0;
  for (fs::directory_iterator I(Twine(RootDir), EC), E; I != E;
       I.increment(EC)) {
    ASSERT_NO_ERROR(EC);
    StringRef DirName = path::filename(I->path());
    EXPECT_TRUE(DirName == "a" || DirName == "b");
    ++Count;
  }
  EXPECT_EQ(Count, 2U);

  ASSERT_NO_ERROR(fs::remove(Twine(RootDir) + "/a"));
  ASSERT_NO_ERROR(fs::remove(Twine(RootDir) + "/b"));
  ASSERT_NO_ERROR(fs::remove(Twine(RootDir)));
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
    raw_fd_ostream File(FilePathname, EC, sys::fs::OF_TextWithCRLF);
    ASSERT_NO_ERROR(EC);
    File << '\n';
  }
  {
    auto Buf = MemoryBuffer::getFile(FilePathname.str());
    EXPECT_TRUE((bool)Buf);
    EXPECT_EQ(Buf.get()->getBuffer(), "\r\n");
  }

  {
    raw_fd_ostream File(FilePathname, EC, sys::fs::OF_None);
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

TEST_F(FileSystemTest, ResizeBeforeMapping) {
  // Create a temp file.
  int FD;
  SmallString<64> TempPath;
  ASSERT_NO_ERROR(fs::createTemporaryFile("prefix", "temp", FD, TempPath));
  ASSERT_NO_ERROR(fs::resize_file_before_mapping_readwrite(FD, 123));

  // Map in temp file. On Windows, fs::resize_file_before_mapping_readwrite is
  // a no-op and the mapping itself will resize the file.
  std::error_code EC;
  {
    fs::mapped_file_region mfr(fs::convertFDToNativeFile(FD),
                               fs::mapped_file_region::readwrite, 123, 0, EC);
    ASSERT_NO_ERROR(EC);
    // Unmap temp file
  }

  // Check the size.
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
  ASSERT_NO_ERROR(
      fs::resize_file_before_mapping_readwrite(FileDescriptor, Size));

  // Map in temp file and add some content
  std::error_code EC;
  StringRef Val("hello there");
  fs::mapped_file_region MaybeMFR;
  EXPECT_FALSE(MaybeMFR);
  {
    fs::mapped_file_region mfr(fs::convertFDToNativeFile(FileDescriptor),
                               fs::mapped_file_region::readwrite, Size, 0, EC);
    ASSERT_NO_ERROR(EC);
    std::copy(Val.begin(), Val.end(), mfr.data());
    // Explicitly add a 0.
    mfr.data()[Val.size()] = 0;

    // Move it out of the scope and confirm mfr is reset.
    MaybeMFR = std::move(mfr);
    EXPECT_FALSE(mfr);
#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
    EXPECT_DEATH(mfr.data(), "Mapping failed but used anyway!");
    EXPECT_DEATH(mfr.size(), "Mapping failed but used anyway!");
#endif
  }

  // Check that the moved-to region is still valid.
  EXPECT_EQ(Val, StringRef(MaybeMFR.data()));
  EXPECT_EQ(Size, MaybeMFR.size());

  // Unmap temp file.
  MaybeMFR.unmap();

  ASSERT_EQ(close(FileDescriptor), 0);

  // Map it back in read-only
  {
    int FD;
    EC = fs::openFileForRead(Twine(TempPath), FD);
    ASSERT_NO_ERROR(EC);
    fs::mapped_file_region mfr(fs::convertFDToNativeFile(FD),
                               fs::mapped_file_region::readonly, Size, 0, EC);
    ASSERT_NO_ERROR(EC);

    // Verify content
    EXPECT_EQ(StringRef(mfr.const_data()), Val);

    // Unmap temp file
    fs::mapped_file_region m(fs::convertFDToNativeFile(FD),
                             fs::mapped_file_region::readonly, Size, 0, EC);
    ASSERT_NO_ERROR(EC);
    ASSERT_EQ(close(FD), 0);
  }
  ASSERT_NO_ERROR(fs::remove(TempPath));
}

TEST(Support, NormalizePath) {
  //                           Input,        Expected Win, Expected Posix
  using TestTuple = std::tuple<const char *, const char *, const char *>;
  std::vector<TestTuple> Tests;
  Tests.emplace_back("a", "a", "a");
  Tests.emplace_back("a/b", "a\\b", "a/b");
  Tests.emplace_back("a\\b", "a\\b", "a/b");
  Tests.emplace_back("a\\\\b", "a\\\\b", "a//b");
  Tests.emplace_back("\\a", "\\a", "/a");
  Tests.emplace_back("a\\", "a\\", "a/");
  Tests.emplace_back("a\\t", "a\\t", "a/t");

  for (auto &T : Tests) {
    SmallString<64> Win(std::get<0>(T));
    SmallString<64> Posix(Win);
    SmallString<64> WinSlash(Win);
    path::native(Win, path::Style::windows);
    path::native(Posix, path::Style::posix);
    path::native(WinSlash, path::Style::windows_slash);
    EXPECT_EQ(std::get<1>(T), Win);
    EXPECT_EQ(std::get<2>(T), Posix);
    EXPECT_EQ(std::get<2>(T), WinSlash);
  }

  for (auto &T : Tests) {
    SmallString<64> WinBackslash(std::get<0>(T));
    SmallString<64> Posix(WinBackslash);
    SmallString<64> WinSlash(WinBackslash);
    path::make_preferred(WinBackslash, path::Style::windows_backslash);
    path::make_preferred(Posix, path::Style::posix);
    path::make_preferred(WinSlash, path::Style::windows_slash);
    EXPECT_EQ(std::get<1>(T), WinBackslash);
    EXPECT_EQ(std::get<0>(T), Posix); // Posix remains unchanged here
    EXPECT_EQ(std::get<2>(T), WinSlash);
  }

#if defined(_WIN32)
  SmallString<64> PathHome;
  path::home_directory(PathHome);

  const char *Path7a = "~/aaa";
  SmallString<64> Path7(Path7a);
  path::native(Path7, path::Style::windows_backslash);
  EXPECT_TRUE(Path7.ends_with("\\aaa"));
  EXPECT_TRUE(Path7.starts_with(PathHome));
  EXPECT_EQ(Path7.size(), PathHome.size() + strlen(Path7a + 1));
  Path7 = Path7a;
  path::native(Path7, path::Style::windows_slash);
  EXPECT_TRUE(Path7.ends_with("/aaa"));
  EXPECT_TRUE(Path7.starts_with(PathHome));
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
  path::native(Path10, path::Style::windows_backslash);
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
  return std::string(buffer.str());
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
  EXPECT_EQ("C:\\a\\c", remove_dots("C:\\foo\\bar//..\\..\\a\\c", true,
                                    path::Style::windows));

  EXPECT_EQ("C:\\bar",
            remove_dots("C:/foo/../bar", true, path::Style::windows));
  EXPECT_EQ("C:\\foo\\bar",
            remove_dots("C:/foo/bar", true, path::Style::windows));
  EXPECT_EQ("C:\\foo\\bar",
            remove_dots("C:/foo\\bar", true, path::Style::windows));
  EXPECT_EQ("\\", remove_dots("/", true, path::Style::windows));
  EXPECT_EQ("C:\\", remove_dots("C:/", true, path::Style::windows));

  // Some clients of remove_dots expect it to remove trailing slashes. Again,
  // this is emergent behavior that VFS relies on, and not inherently part of
  // the specification.
  EXPECT_EQ("C:\\foo\\bar",
            remove_dots("C:\\foo\\bar\\", true, path::Style::windows));
  EXPECT_EQ("/foo/bar",
            remove_dots("/foo/bar/", true, path::Style::posix));

  // A double separator is rewritten.
  EXPECT_EQ("C:\\foo\\bar",
            remove_dots("C:/foo//bar", true, path::Style::windows));

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
  EXPECT_EQ("/", remove_dots("/", true, path::Style::posix));

  // FIXME: Leaving behind this double leading slash seems like a bug.
  EXPECT_EQ("//foo/bar",
            remove_dots("//foo/bar/", true, path::Style::posix));

  SmallString<64> Path2("././c");
  EXPECT_TRUE(path::remove_dots(Path2, true, path::Style::posix));
  EXPECT_EQ("c", Path2);
}

TEST(Support, ReplacePathPrefix) {
  SmallString<64> Path1("/foo");
  SmallString<64> Path2("/old/foo");
  SmallString<64> Path3("/oldnew/foo");
  SmallString<64> Path4("C:\\old/foo\\bar");
  SmallString<64> OldPrefix("/old");
  SmallString<64> OldPrefixSep("/old/");
  SmallString<64> OldPrefixWin("c:/oLD/F");
  SmallString<64> NewPrefix("/new");
  SmallString<64> NewPrefix2("/longernew");
  SmallString<64> EmptyPrefix("");
  bool Found;

  SmallString<64> Path = Path1;
  Found = path::replace_path_prefix(Path, OldPrefix, NewPrefix);
  EXPECT_FALSE(Found);
  EXPECT_EQ(Path, "/foo");
  Path = Path2;
  Found = path::replace_path_prefix(Path, OldPrefix, NewPrefix);
  EXPECT_TRUE(Found);
  EXPECT_EQ(Path, "/new/foo");
  Path = Path2;
  Found = path::replace_path_prefix(Path, OldPrefix, NewPrefix2);
  EXPECT_TRUE(Found);
  EXPECT_EQ(Path, "/longernew/foo");
  Path = Path1;
  Found = path::replace_path_prefix(Path, EmptyPrefix, NewPrefix);
  EXPECT_TRUE(Found);
  EXPECT_EQ(Path, "/new/foo");
  Path = Path2;
  Found = path::replace_path_prefix(Path, OldPrefix, EmptyPrefix);
  EXPECT_TRUE(Found);
  EXPECT_EQ(Path, "/foo");
  Path = Path2;
  Found = path::replace_path_prefix(Path, OldPrefixSep, EmptyPrefix);
  EXPECT_TRUE(Found);
  EXPECT_EQ(Path, "foo");
  Path = Path3;
  Found = path::replace_path_prefix(Path, OldPrefix, NewPrefix);
  EXPECT_TRUE(Found);
  EXPECT_EQ(Path, "/newnew/foo");
  Path = Path3;
  Found = path::replace_path_prefix(Path, OldPrefix, NewPrefix2);
  EXPECT_TRUE(Found);
  EXPECT_EQ(Path, "/longernewnew/foo");
  Path = Path1;
  Found = path::replace_path_prefix(Path, EmptyPrefix, NewPrefix);
  EXPECT_TRUE(Found);
  EXPECT_EQ(Path, "/new/foo");
  Path = OldPrefix;
  Found = path::replace_path_prefix(Path, OldPrefix, NewPrefix);
  EXPECT_TRUE(Found);
  EXPECT_EQ(Path, "/new");
  Path = OldPrefixSep;
  Found = path::replace_path_prefix(Path, OldPrefix, NewPrefix);
  EXPECT_TRUE(Found);
  EXPECT_EQ(Path, "/new/");
  Path = OldPrefix;
  Found = path::replace_path_prefix(Path, OldPrefixSep, NewPrefix);
  EXPECT_FALSE(Found);
  EXPECT_EQ(Path, "/old");
  Path = Path4;
  Found = path::replace_path_prefix(Path, OldPrefixWin, NewPrefix,
                                    path::Style::windows);
  EXPECT_TRUE(Found);
  EXPECT_EQ(Path, "/newoo\\bar");
  Path = Path4;
  Found = path::replace_path_prefix(Path, OldPrefixWin, NewPrefix,
                                    path::Style::posix);
  EXPECT_FALSE(Found);
  EXPECT_EQ(Path, "C:\\old/foo\\bar");
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
  ASSERT_NO_ERROR(fs::setLastAccessAndModificationTime(FileDescriptor, Epoch));
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

TEST_F(FileSystemTest, OpenDirectoryAsFileForRead) {
  std::string Buf(5, '?');
  Expected<fs::file_t> FD = fs::openNativeFileForRead(TestDirectory);
#ifdef _WIN32
  EXPECT_EQ(errorToErrorCode(FD.takeError()), errc::is_a_directory);
#else
  ASSERT_THAT_EXPECTED(FD, Succeeded());
  auto Close = make_scope_exit([&] { fs::closeFile(*FD); });
  Expected<size_t> BytesRead =
      fs::readNativeFile(*FD, MutableArrayRef(&*Buf.begin(), Buf.size()));
  EXPECT_EQ(errorToErrorCode(BytesRead.takeError()), errc::is_a_directory);
#endif
}

TEST_F(FileSystemTest, OpenDirectoryAsFileForWrite) {
  int FD;
  std::error_code EC = fs::openFileForWrite(Twine(TestDirectory), FD);
  if (!EC)
    ::close(FD);
  EXPECT_EQ(EC, errc::is_a_directory);
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
  std::optional<FileDescriptorCloser> Closer;

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
  std::optional<FileDescriptorCloser> Closer;

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
  // hack that shouldn't work, but is left for compatibility.  OF_Append
  // overrides
  // the specified disposition.
  for (fs::CreationDisposition Disp : Disps) {
    int FD;
    std::optional<FileDescriptorCloser> Closer;

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

TEST_F(FileSystemTest, readNativeFile) {
  createFileWithData(NonExistantFile, false, fs::CD_CreateNew, "01234");
  FileRemover Cleanup(NonExistantFile);
  const auto &Read = [&](size_t ToRead) -> Expected<std::string> {
    std::string Buf(ToRead, '?');
    Expected<fs::file_t> FD = fs::openNativeFileForRead(NonExistantFile);
    if (!FD)
      return FD.takeError();
    auto Close = make_scope_exit([&] { fs::closeFile(*FD); });
    if (Expected<size_t> BytesRead = fs::readNativeFile(
            *FD, MutableArrayRef(&*Buf.begin(), Buf.size())))
      return Buf.substr(0, *BytesRead);
    else
      return BytesRead.takeError();
  };
  EXPECT_THAT_EXPECTED(Read(5), HasValue("01234"));
  EXPECT_THAT_EXPECTED(Read(3), HasValue("012"));
  EXPECT_THAT_EXPECTED(Read(6), HasValue("01234"));
}

TEST_F(FileSystemTest, readNativeFileToEOF) {
  constexpr StringLiteral Content = "0123456789";
  createFileWithData(NonExistantFile, false, fs::CD_CreateNew, Content);
  FileRemover Cleanup(NonExistantFile);
  const auto &Read = [&](SmallVectorImpl<char> &V,
                         std::optional<ssize_t> ChunkSize) {
    Expected<fs::file_t> FD = fs::openNativeFileForRead(NonExistantFile);
    if (!FD)
      return FD.takeError();
    auto Close = make_scope_exit([&] { fs::closeFile(*FD); });
    if (ChunkSize)
      return fs::readNativeFileToEOF(*FD, V, *ChunkSize);
    return fs::readNativeFileToEOF(*FD, V);
  };

  // Check basic operation.
  {
    SmallString<0> NoSmall;
    SmallString<fs::DefaultReadChunkSize + Content.size()> StaysSmall;
    SmallVectorImpl<char> *Vectors[] = {
        static_cast<SmallVectorImpl<char> *>(&NoSmall),
        static_cast<SmallVectorImpl<char> *>(&StaysSmall),
    };
    for (SmallVectorImpl<char> *V : Vectors) {
      ASSERT_THAT_ERROR(Read(*V, std::nullopt), Succeeded());
      ASSERT_EQ(Content, StringRef(V->begin(), V->size()));
    }
    ASSERT_EQ(fs::DefaultReadChunkSize + Content.size(), StaysSmall.capacity());

    // Check appending.
    {
      constexpr StringLiteral Prefix = "prefix-";
      for (SmallVectorImpl<char> *V : Vectors) {
        V->assign(Prefix.begin(), Prefix.end());
        ASSERT_THAT_ERROR(Read(*V, std::nullopt), Succeeded());
        ASSERT_EQ((Prefix + Content).str(), StringRef(V->begin(), V->size()));
      }
    }
  }

  // Check that the chunk size (if specified) is respected.
  SmallString<Content.size() + 5> SmallChunks;
  ASSERT_THAT_ERROR(Read(SmallChunks, 5), Succeeded());
  ASSERT_EQ(SmallChunks, Content);
  ASSERT_EQ(Content.size() + 5, SmallChunks.capacity());
}

TEST_F(FileSystemTest, readNativeFileSlice) {
  createFileWithData(NonExistantFile, false, fs::CD_CreateNew, "01234");
  FileRemover Cleanup(NonExistantFile);
  Expected<fs::file_t> FD = fs::openNativeFileForRead(NonExistantFile);
  ASSERT_THAT_EXPECTED(FD, Succeeded());
  auto Close = make_scope_exit([&] { fs::closeFile(*FD); });
  const auto &Read = [&](size_t Offset,
                         size_t ToRead) -> Expected<std::string> {
    std::string Buf(ToRead, '?');
    if (Expected<size_t> BytesRead = fs::readNativeFileSlice(
            *FD, MutableArrayRef(&*Buf.begin(), Buf.size()), Offset))
      return Buf.substr(0, *BytesRead);
    else
      return BytesRead.takeError();
  };
  EXPECT_THAT_EXPECTED(Read(0, 5), HasValue("01234"));
  EXPECT_THAT_EXPECTED(Read(0, 3), HasValue("012"));
  EXPECT_THAT_EXPECTED(Read(2, 3), HasValue("234"));
  EXPECT_THAT_EXPECTED(Read(0, 6), HasValue("01234"));
  EXPECT_THAT_EXPECTED(Read(2, 6), HasValue("234"));
  EXPECT_THAT_EXPECTED(Read(5, 5), HasValue(""));
}

TEST_F(FileSystemTest, is_local) {
  bool TestDirectoryIsLocal;
  ASSERT_NO_ERROR(fs::is_local(TestDirectory, TestDirectoryIsLocal));
  EXPECT_EQ(TestDirectoryIsLocal, fs::is_local(TestDirectory));

  int FD;
  SmallString<128> TempPath;
  ASSERT_NO_ERROR(
      fs::createUniqueFile(Twine(TestDirectory) + "/temp", FD, TempPath));
  FileRemover Cleanup(TempPath);

  // Make sure it exists.
  ASSERT_TRUE(sys::fs::exists(Twine(TempPath)));

  bool TempFileIsLocal;
  ASSERT_NO_ERROR(fs::is_local(FD, TempFileIsLocal));
  EXPECT_EQ(TempFileIsLocal, fs::is_local(FD));
  ::close(FD);

  // Expect that the file and its parent directory are equally local or equally
  // remote.
  EXPECT_EQ(TestDirectoryIsLocal, TempFileIsLocal);
}

TEST_F(FileSystemTest, getUmask) {
#ifdef _WIN32
  EXPECT_EQ(fs::getUmask(), 0U) << "Should always be 0 on Windows.";
#else
  unsigned OldMask = ::umask(0022);
  unsigned CurrentMask = fs::getUmask();
  EXPECT_EQ(CurrentMask, 0022U)
      << "getUmask() didn't return previously set umask()";
  EXPECT_EQ(::umask(OldMask), mode_t(0022U))
      << "getUmask() may have changed umask()";
#endif
}

TEST_F(FileSystemTest, RespectUmask) {
#ifndef _WIN32
  unsigned OldMask = ::umask(0022);

  int FD;
  SmallString<128> TempPath;
  ASSERT_NO_ERROR(fs::createTemporaryFile("prefix", "temp", FD, TempPath));

  fs::perms AllRWE = static_cast<fs::perms>(0777);

  ASSERT_NO_ERROR(fs::setPermissions(TempPath, AllRWE));

  ErrorOr<fs::perms> Perms = fs::getPermissions(TempPath);
  ASSERT_TRUE(!!Perms);
  EXPECT_EQ(Perms.get(), AllRWE) << "Should have ignored umask by default";

  ASSERT_NO_ERROR(fs::setPermissions(TempPath, AllRWE));

  Perms = fs::getPermissions(TempPath);
  ASSERT_TRUE(!!Perms);
  EXPECT_EQ(Perms.get(), AllRWE) << "Should have ignored umask";

  ASSERT_NO_ERROR(
      fs::setPermissions(FD, static_cast<fs::perms>(AllRWE & ~fs::getUmask())));
  Perms = fs::getPermissions(TempPath);
  ASSERT_TRUE(!!Perms);
  EXPECT_EQ(Perms.get(), static_cast<fs::perms>(0755))
      << "Did not respect umask";

  (void)::umask(0057);

  ASSERT_NO_ERROR(
      fs::setPermissions(FD, static_cast<fs::perms>(AllRWE & ~fs::getUmask())));
  Perms = fs::getPermissions(TempPath);
  ASSERT_TRUE(!!Perms);
  EXPECT_EQ(Perms.get(), static_cast<fs::perms>(0720))
      << "Did not respect umask";

  (void)::umask(OldMask);
  (void)::close(FD);
#endif
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
  // AIX and Solaris without root will mask off (i.e., lose) the sticky bit
  // on files.
#if !defined(__FreeBSD__) && !defined(__NetBSD__) && !defined(__OpenBSD__) &&  \
    !defined(_AIX) && !(defined(__sun__) && defined(__svr4__))
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
#endif // !FreeBSD && !NetBSD && !OpenBSD && !AIX

  EXPECT_EQ(fs::setPermissions(TempPath, fs::all_perms & ~fs::sticky_bit),
                               NoError);
  EXPECT_TRUE(CheckPermissions(fs::all_perms & ~fs::sticky_bit));
#endif
}

#ifdef _WIN32
TEST_F(FileSystemTest, widenPath) {
  const std::wstring LongPathPrefix(L"\\\\?\\");

  // Test that the length limit is checked against the UTF-16 length and not the
  // UTF-8 length.
  std::string Input("C:\\foldername\\");
  const std::string Pi("\xcf\x80"); // UTF-8 lower case pi.
  // Add Pi up to the MAX_PATH limit.
  const size_t NumChars = MAX_PATH - Input.size() - 1;
  for (size_t i = 0; i < NumChars; ++i)
    Input += Pi;
  // Check that UTF-8 length already exceeds MAX_PATH.
  EXPECT_GT(Input.size(), (size_t)MAX_PATH);
  SmallVector<wchar_t, MAX_PATH + 16> Result;
  ASSERT_NO_ERROR(windows::widenPath(Input, Result));
  // Result should not start with the long path prefix.
  EXPECT_TRUE(std::wmemcmp(Result.data(), LongPathPrefix.c_str(),
                           LongPathPrefix.size()) != 0);
  EXPECT_EQ(Result.size(), (size_t)MAX_PATH - 1);

  // Add another Pi to exceed the MAX_PATH limit.
  Input += Pi;
  // Construct the expected result.
  SmallVector<wchar_t, MAX_PATH + 16> Expected;
  ASSERT_NO_ERROR(windows::UTF8ToUTF16(Input, Expected));
  Expected.insert(Expected.begin(), LongPathPrefix.begin(),
                  LongPathPrefix.end());

  ASSERT_NO_ERROR(windows::widenPath(Input, Result));
  EXPECT_EQ(Result, Expected);
  // Pass a path with forward slashes, check that it ends up with
  // backslashes when widened with the long path prefix.
  SmallString<MAX_PATH + 16> InputForward(Input);
  path::make_preferred(InputForward, path::Style::windows_slash);
  ASSERT_NO_ERROR(windows::widenPath(InputForward, Result));
  EXPECT_EQ(Result, Expected);

  // Pass a path which has the long path prefix prepended originally, but
  // which is short enough to not require the long path prefix. If such a
  // path is passed with forward slashes, make sure it gets normalized to
  // backslashes.
  SmallString<MAX_PATH + 16> PrefixedPath("\\\\?\\C:\\foldername");
  ASSERT_NO_ERROR(windows::UTF8ToUTF16(PrefixedPath, Expected));
  // Mangle the input to forward slashes.
  path::make_preferred(PrefixedPath, path::Style::windows_slash);
  ASSERT_NO_ERROR(windows::widenPath(PrefixedPath, Result));
  EXPECT_EQ(Result, Expected);

  // A short path with an inconsistent prefix is passed through as-is; this
  // is a degenerate case that we currently don't care about handling.
  PrefixedPath.assign("/\\?/C:/foldername");
  ASSERT_NO_ERROR(windows::UTF8ToUTF16(PrefixedPath, Expected));
  ASSERT_NO_ERROR(windows::widenPath(PrefixedPath, Result));
  EXPECT_EQ(Result, Expected);

  // Test that UNC paths are handled correctly.
  const std::string ShareName("\\\\sharename\\");
  const std::string FileName("\\filename");
  // Initialize directory name so that the input is within the MAX_PATH limit.
  const char DirChar = 'x';
  std::string DirName(MAX_PATH - ShareName.size() - FileName.size() - 1,
                      DirChar);

  Input = ShareName + DirName + FileName;
  ASSERT_NO_ERROR(windows::widenPath(Input, Result));
  // Result should not start with the long path prefix.
  EXPECT_TRUE(std::wmemcmp(Result.data(), LongPathPrefix.c_str(),
                           LongPathPrefix.size()) != 0);
  EXPECT_EQ(Result.size(), (size_t)MAX_PATH - 1);

  // Extend the directory name so the input exceeds the MAX_PATH limit.
  DirName += DirChar;
  Input = ShareName + DirName + FileName;
  // Construct the expected result.
  ASSERT_NO_ERROR(windows::UTF8ToUTF16(StringRef(Input).substr(2), Expected));
  const std::wstring UNCPrefix(LongPathPrefix + L"UNC\\");
  Expected.insert(Expected.begin(), UNCPrefix.begin(), UNCPrefix.end());

  ASSERT_NO_ERROR(windows::widenPath(Input, Result));
  EXPECT_EQ(Result, Expected);

  // Check that Unix separators are handled correctly.
  std::replace(Input.begin(), Input.end(), '\\', '/');
  ASSERT_NO_ERROR(windows::widenPath(Input, Result));
  EXPECT_EQ(Result, Expected);

  // Check the removal of "dots".
  Input = ShareName + DirName + "\\.\\foo\\.\\.." + FileName;
  ASSERT_NO_ERROR(windows::widenPath(Input, Result));
  EXPECT_EQ(Result, Expected);
}
#endif

#ifdef _WIN32
// Windows refuses lock request if file region is already locked by the same
// process. POSIX system in this case updates the existing lock.
TEST_F(FileSystemTest, FileLocker) {
  using namespace std::chrono;
  int FD;
  std::error_code EC;
  SmallString<64> TempPath;
  EC = fs::createTemporaryFile("test", "temp", FD, TempPath);
  ASSERT_NO_ERROR(EC);
  FileRemover Cleanup(TempPath);
  raw_fd_ostream Stream(TempPath, EC);

  EC = fs::tryLockFile(FD);
  ASSERT_NO_ERROR(EC);
  EC = fs::unlockFile(FD);
  ASSERT_NO_ERROR(EC);

  if (auto L = Stream.lock()) {
    ASSERT_ERROR(fs::tryLockFile(FD));
    ASSERT_NO_ERROR(L->unlock());
    ASSERT_NO_ERROR(fs::tryLockFile(FD));
    ASSERT_NO_ERROR(fs::unlockFile(FD));
  } else {
    ADD_FAILURE();
    handleAllErrors(L.takeError(), [&](ErrorInfoBase &EIB) {});
  }

  ASSERT_NO_ERROR(fs::tryLockFile(FD));
  ASSERT_NO_ERROR(fs::unlockFile(FD));

  {
    Expected<fs::FileLocker> L1 = Stream.lock();
    ASSERT_THAT_EXPECTED(L1, Succeeded());
    raw_fd_ostream Stream2(FD, false);
    Expected<fs::FileLocker> L2 = Stream2.tryLockFor(250ms);
    ASSERT_THAT_EXPECTED(L2, Failed());
    ASSERT_NO_ERROR(L1->unlock());
    Expected<fs::FileLocker> L3 = Stream.tryLockFor(0ms);
    ASSERT_THAT_EXPECTED(L3, Succeeded());
  }

  ASSERT_NO_ERROR(fs::tryLockFile(FD));
  ASSERT_NO_ERROR(fs::unlockFile(FD));
}
#endif

TEST_F(FileSystemTest, CopyFile) {
  unittest::TempDir RootTestDirectory("CopyFileTest", /*Unique=*/true);

  SmallVector<std::string> Data;
  SmallVector<SmallString<128>> Sources;
  for (int I = 0, E = 3; I != E; ++I) {
    Data.push_back(Twine(I).str());
    Sources.emplace_back(RootTestDirectory.path());
    path::append(Sources.back(), "source" + Data.back() + ".txt");
    createFileWithData(Sources.back(), /*ShouldExistBefore=*/false,
                       fs::CD_CreateNew, Data.back());
  }

  // Copy the first file to a non-existing file.
  SmallString<128> Destination(RootTestDirectory.path());
  path::append(Destination, "destination");
  ASSERT_FALSE(fs::exists(Destination));
  fs::copy_file(Sources[0], Destination);
  verifyFileContents(Destination, Data[0]);

  // Copy the second file to an existing file.
  fs::copy_file(Sources[1], Destination);
  verifyFileContents(Destination, Data[1]);

  // Note: The remaining logic is targeted at a potential failure case related
  // to file cloning and symlinks on Darwin. On Windows, fs::create_link() does
  // not return success here so the test is skipped.
#if !defined(_WIN32)
  // Set up a symlink to the third file.
  SmallString<128> Symlink(RootTestDirectory.path());
  path::append(Symlink, "symlink");
  ASSERT_NO_ERROR(fs::create_link(path::filename(Sources[2]), Symlink));
  verifyFileContents(Symlink, Data[2]);

  // fs::getUniqueID() should follow symlinks. Otherwise, this isn't good test
  // coverage.
  fs::UniqueID SymlinkID;
  fs::UniqueID Data2ID;
  ASSERT_NO_ERROR(fs::getUniqueID(Symlink, SymlinkID));
  ASSERT_NO_ERROR(fs::getUniqueID(Sources[2], Data2ID));
  ASSERT_EQ(SymlinkID, Data2ID);

  // Copy the third file through the symlink.
  fs::copy_file(Symlink, Destination);
  verifyFileContents(Destination, Data[2]);

  // Confirm the destination is not a link to the original file, and not a
  // symlink.
  bool IsDestinationSymlink;
  ASSERT_NO_ERROR(fs::is_symlink_file(Destination, IsDestinationSymlink));
  ASSERT_FALSE(IsDestinationSymlink);
  fs::UniqueID DestinationID;
  ASSERT_NO_ERROR(fs::getUniqueID(Destination, DestinationID));
  ASSERT_NE(SymlinkID, DestinationID);
#endif
}

} // anonymous namespace
