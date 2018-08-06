//===-- TestCompletion.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Utility/StringList.h"
#include "lldb/Utility/TildeExpressionResolver.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "TestingSupport/MockTildeExpressionResolver.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;
using namespace llvm;
using namespace lldb_private;

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

class CompletionTest : public testing::Test {
protected:
  /// Unique temporary directory in which all created filesystem entities must
  /// be placed. It is removed at the end of the test suite.
  static SmallString<128> BaseDir;

  static SmallString<128> OriginalWorkingDir;

  static SmallString<128> DirFoo;
  static SmallString<128> DirFooA;
  static SmallString<128> DirFooB;
  static SmallString<128> DirFooC;
  static SmallString<128> DirBar;
  static SmallString<128> DirBaz;
  static SmallString<128> DirTestFolder;
  static SmallString<128> DirNested;

  static SmallString<128> FileAA;
  static SmallString<128> FileAB;
  static SmallString<128> FileAC;
  static SmallString<128> FileFoo;
  static SmallString<128> FileBar;
  static SmallString<128> FileBaz;

  void SetUp() override { llvm::sys::fs::set_current_path(OriginalWorkingDir); }

  static void SetUpTestCase() {
    llvm::sys::fs::current_path(OriginalWorkingDir);

    ASSERT_NO_ERROR(fs::createUniqueDirectory("FsCompletion", BaseDir));
    const char *DirNames[] = {"foo", "fooa", "foob",        "fooc",
                              "bar", "baz",  "test_folder", "foo/nested"};
    const char *FileNames[] = {"aa1234.tmp",  "ab1234.tmp",  "ac1234.tmp",
                               "foo1234.tmp", "bar1234.tmp", "baz1234.tmp"};
    SmallString<128> *Dirs[] = {&DirFoo, &DirFooA, &DirFooB,       &DirFooC,
                                &DirBar, &DirBaz,  &DirTestFolder, &DirNested};
    for (auto Dir : llvm::zip(DirNames, Dirs)) {
      auto &Path = *std::get<1>(Dir);
      Path = BaseDir;
      path::append(Path, std::get<0>(Dir));
      ASSERT_NO_ERROR(fs::create_directories(Path));
    }

    SmallString<128> *Files[] = {&FileAA,  &FileAB,  &FileAC,
                                 &FileFoo, &FileBar, &FileBaz};
    for (auto File : llvm::zip(FileNames, Files)) {
      auto &Path = *std::get<1>(File);
      Path = BaseDir;
      path::append(Path, std::get<0>(File));
      int FD;
      ASSERT_NO_ERROR(fs::createUniqueFile(Path, FD, Path));
      ::close(FD);
    }
  }

  static void TearDownTestCase() {
    ASSERT_NO_ERROR(fs::remove_directories(BaseDir));
  }

  static bool HasEquivalentFile(const Twine &Path, const StringList &Paths) {
    for (size_t I = 0; I < Paths.GetSize(); ++I) {
      if (fs::equivalent(Path, Paths[I]))
        return true;
    }
    return false;
  }

  void DoDirCompletions(const Twine &Prefix,
                        StandardTildeExpressionResolver &Resolver,
                        StringList &Results) {
    // When a partial name matches, it returns all matches.  If it matches both
    // a full name AND some partial names, it returns all of them.
    uint32_t Count =
        CommandCompletions::DiskDirectories(Prefix + "foo", Results, Resolver);
    ASSERT_EQ(4u, Count);
    ASSERT_EQ(Count, Results.GetSize());
    EXPECT_TRUE(HasEquivalentFile(DirFoo, Results));
    EXPECT_TRUE(HasEquivalentFile(DirFooA, Results));
    EXPECT_TRUE(HasEquivalentFile(DirFooB, Results));
    EXPECT_TRUE(HasEquivalentFile(DirFooC, Results));

    // If it matches only partial names, it still works as expected.
    Count = CommandCompletions::DiskDirectories(Twine(Prefix) + "b", Results,
                                                Resolver);
    ASSERT_EQ(2u, Count);
    ASSERT_EQ(Count, Results.GetSize());
    EXPECT_TRUE(HasEquivalentFile(DirBar, Results));
    EXPECT_TRUE(HasEquivalentFile(DirBaz, Results));
  }
};

SmallString<128> CompletionTest::BaseDir;
SmallString<128> CompletionTest::OriginalWorkingDir;

SmallString<128> CompletionTest::DirFoo;
SmallString<128> CompletionTest::DirFooA;
SmallString<128> CompletionTest::DirFooB;
SmallString<128> CompletionTest::DirFooC;
SmallString<128> CompletionTest::DirBar;
SmallString<128> CompletionTest::DirBaz;
SmallString<128> CompletionTest::DirTestFolder;
SmallString<128> CompletionTest::DirNested;

SmallString<128> CompletionTest::FileAA;
SmallString<128> CompletionTest::FileAB;
SmallString<128> CompletionTest::FileAC;
SmallString<128> CompletionTest::FileFoo;
SmallString<128> CompletionTest::FileBar;
SmallString<128> CompletionTest::FileBaz;
}

static std::vector<std::string> toVector(const StringList &SL) {
  std::vector<std::string> Result;
  for (size_t Idx = 0; Idx < SL.GetSize(); ++Idx)
    Result.push_back(SL[Idx]);
  return Result;
}
using testing::UnorderedElementsAre;

TEST_F(CompletionTest, DirCompletionAbsolute) {
  // All calls to DiskDirectories() return only directories, even when
  // there are files which also match.  The tests below all check this
  // by asserting an exact result count, and verifying against known
  // folders.

  std::string Prefixes[] = {(Twine(BaseDir) + "/").str(), ""};

  StandardTildeExpressionResolver Resolver;
  StringList Results;

  // When a directory is specified that doesn't end in a slash, it searches
  // for that directory, not items under it.
  size_t Count =
      CommandCompletions::DiskDirectories(BaseDir, Results, Resolver);
  ASSERT_EQ(1u, Count);
  ASSERT_EQ(Count, Results.GetSize());
  EXPECT_TRUE(HasEquivalentFile(BaseDir, Results));

  // When the same directory ends with a slash, it finds all children.
  Count = CommandCompletions::DiskDirectories(Prefixes[0], Results, Resolver);
  ASSERT_EQ(7u, Count);
  ASSERT_EQ(Count, Results.GetSize());
  EXPECT_TRUE(HasEquivalentFile(DirFoo, Results));
  EXPECT_TRUE(HasEquivalentFile(DirFooA, Results));
  EXPECT_TRUE(HasEquivalentFile(DirFooB, Results));
  EXPECT_TRUE(HasEquivalentFile(DirFooC, Results));
  EXPECT_TRUE(HasEquivalentFile(DirBar, Results));
  EXPECT_TRUE(HasEquivalentFile(DirBaz, Results));
  EXPECT_TRUE(HasEquivalentFile(DirTestFolder, Results));

  DoDirCompletions(Twine(BaseDir) + "/", Resolver, Results);
  llvm::sys::fs::set_current_path(BaseDir);
  DoDirCompletions("", Resolver, Results);
}

TEST_F(CompletionTest, FileCompletionAbsolute) {
  // All calls to DiskFiles() return both files and directories  The tests below
  // all check this by asserting an exact result count, and verifying against
  // known folders.

  StandardTildeExpressionResolver Resolver;
  StringList Results;
  // When an item is specified that doesn't end in a slash but exactly matches
  // one item, it returns that item.
  size_t Count = CommandCompletions::DiskFiles(Twine(BaseDir) + "/fooa",
                                               Results, Resolver);
  ASSERT_EQ(1u, Count);
  ASSERT_EQ(Count, Results.GetSize());
  EXPECT_TRUE(HasEquivalentFile(DirFooA, Results));

  // The previous check verified a directory match.  But it should work for
  // files too.
  Count =
      CommandCompletions::DiskFiles(Twine(BaseDir) + "/aa", Results, Resolver);
  ASSERT_EQ(1u, Count);
  ASSERT_EQ(Count, Results.GetSize());
  EXPECT_TRUE(HasEquivalentFile(FileAA, Results));

  // When it ends with a slash, it should find all files and directories.
  Count =
      CommandCompletions::DiskFiles(Twine(BaseDir) + "/", Results, Resolver);
  ASSERT_EQ(13u, Count);
  ASSERT_EQ(Count, Results.GetSize());
  EXPECT_TRUE(HasEquivalentFile(DirFoo, Results));
  EXPECT_TRUE(HasEquivalentFile(DirFooA, Results));
  EXPECT_TRUE(HasEquivalentFile(DirFooB, Results));
  EXPECT_TRUE(HasEquivalentFile(DirFooC, Results));
  EXPECT_TRUE(HasEquivalentFile(DirBar, Results));
  EXPECT_TRUE(HasEquivalentFile(DirBaz, Results));
  EXPECT_TRUE(HasEquivalentFile(DirTestFolder, Results));

  EXPECT_TRUE(HasEquivalentFile(FileAA, Results));
  EXPECT_TRUE(HasEquivalentFile(FileAB, Results));
  EXPECT_TRUE(HasEquivalentFile(FileAC, Results));
  EXPECT_TRUE(HasEquivalentFile(FileFoo, Results));
  EXPECT_TRUE(HasEquivalentFile(FileBar, Results));
  EXPECT_TRUE(HasEquivalentFile(FileBaz, Results));

  // When a partial name matches, it returns all file & directory matches.
  Count =
      CommandCompletions::DiskFiles(Twine(BaseDir) + "/foo", Results, Resolver);
  ASSERT_EQ(5u, Count);
  ASSERT_EQ(Count, Results.GetSize());
  EXPECT_TRUE(HasEquivalentFile(DirFoo, Results));
  EXPECT_TRUE(HasEquivalentFile(DirFooA, Results));
  EXPECT_TRUE(HasEquivalentFile(DirFooB, Results));
  EXPECT_TRUE(HasEquivalentFile(DirFooC, Results));
  EXPECT_TRUE(HasEquivalentFile(FileFoo, Results));
}

TEST_F(CompletionTest, DirCompletionUsername) {
  MockTildeExpressionResolver Resolver("James", BaseDir);
  Resolver.AddKnownUser("Kirk", DirFooB);
  Resolver.AddKnownUser("Lars", DirFooC);
  Resolver.AddKnownUser("Jason", DirFoo);
  Resolver.AddKnownUser("Larry", DirFooA);
  std::string sep = path::get_separator();

  // Just resolving current user's home directory by itself should return the
  // directory.
  StringList Results;
  size_t Count = CommandCompletions::DiskDirectories("~", Results, Resolver);
  EXPECT_EQ(Count, Results.GetSize());
  EXPECT_THAT(toVector(Results), UnorderedElementsAre("~" + sep));

  // With a slash appended, it should return all items in the directory.
  Count = CommandCompletions::DiskDirectories("~/", Results, Resolver);
  EXPECT_THAT(toVector(Results),
              UnorderedElementsAre(
                  "~/foo" + sep, "~/fooa" + sep, "~/foob" + sep, "~/fooc" + sep,
                  "~/bar" + sep, "~/baz" + sep, "~/test_folder" + sep));
  EXPECT_EQ(Count, Results.GetSize());

  // Check that we can complete directories in nested paths
  Count = CommandCompletions::DiskDirectories("~/foo/", Results, Resolver);
  EXPECT_EQ(Count, Results.GetSize());
  EXPECT_THAT(toVector(Results), UnorderedElementsAre("~/foo/nested" + sep));

  Count = CommandCompletions::DiskDirectories("~/foo/nes", Results, Resolver);
  EXPECT_EQ(Count, Results.GetSize());
  EXPECT_THAT(toVector(Results), UnorderedElementsAre("~/foo/nested" + sep));

  // With ~username syntax it should return one match if there is an exact
  // match.  It shouldn't translate to the actual directory, it should keep the
  // form the user typed.
  Count = CommandCompletions::DiskDirectories("~Lars", Results, Resolver);
  EXPECT_EQ(Count, Results.GetSize());
  EXPECT_THAT(toVector(Results), UnorderedElementsAre("~Lars" + sep));

  // But with a username that is not found, no results are returned.
  Count = CommandCompletions::DiskDirectories("~Dave", Results, Resolver);
  EXPECT_EQ(Count, Results.GetSize());
  EXPECT_THAT(toVector(Results), UnorderedElementsAre());

  // And if there are multiple matches, it should return all of them.
  Count = CommandCompletions::DiskDirectories("~La", Results, Resolver);
  EXPECT_EQ(Count, Results.GetSize());
  EXPECT_THAT(toVector(Results),
              UnorderedElementsAre("~Lars" + sep, "~Larry" + sep));
}
