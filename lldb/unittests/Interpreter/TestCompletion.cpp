//===-- TestCompletion.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileSystem.h"
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
  SmallString<128> BaseDir;

  /// The working directory that we got when starting the test. Every test
  /// should chdir into this directory first because some tests maybe chdir
  /// into another one during their run.
  static SmallString<128> OriginalWorkingDir;

  SmallString<128> DirFoo;
  SmallString<128> DirFooA;
  SmallString<128> DirFooB;
  SmallString<128> DirFooC;
  SmallString<128> DirBar;
  SmallString<128> DirBaz;
  SmallString<128> DirTestFolder;
  SmallString<128> DirNested;

  SmallString<128> FileAA;
  SmallString<128> FileAB;
  SmallString<128> FileAC;
  SmallString<128> FileFoo;
  SmallString<128> FileBar;
  SmallString<128> FileBaz;

  void SetUp() override {
    FileSystem::Initialize();

    // chdir back into the original working dir this test binary started with.
    // A previous test may have have changed the working dir.
    ASSERT_NO_ERROR(fs::set_current_path(OriginalWorkingDir));

    // Get the name of the current test. To prevent that by chance two tests
    // get the same temporary directory if createUniqueDirectory fails.
    auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    ASSERT_TRUE(test_info != nullptr);
    std::string name = test_info->name();
    ASSERT_NO_ERROR(fs::createUniqueDirectory("FsCompletion-" + name, BaseDir));

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

  static void SetUpTestCase() {
    ASSERT_NO_ERROR(fs::current_path(OriginalWorkingDir));
  }

  void TearDown() override {
    ASSERT_NO_ERROR(fs::remove_directories(BaseDir));
    FileSystem::Terminate();
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

SmallString<128> CompletionTest::OriginalWorkingDir;
} // namespace

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
  // Sanity check that the path we complete on exists and isn't too long.
  size_t Count = CommandCompletions::DiskDirectories(Twine(BaseDir) + "/fooa",
                                                     Results, Resolver);
  ASSERT_EQ(1u, Count);
  ASSERT_EQ(Count, Results.GetSize());
  EXPECT_TRUE(HasEquivalentFile(DirFooA, Results));

  Count = CommandCompletions::DiskDirectories(Twine(BaseDir) + "/.", Results,
                                              Resolver);
  ASSERT_EQ(0u, Count);
  ASSERT_EQ(Count, Results.GetSize());

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
