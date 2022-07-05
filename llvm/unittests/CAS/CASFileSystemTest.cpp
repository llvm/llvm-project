//===- CASFileSystemTest.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASFileSystem.h"
#include "llvm/CAS/CASDB.h"
#include "llvm/CAS/HierarchicalTreeBuilder.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

template <class T>
static std::unique_ptr<T>
errorOrToPointer(ErrorOr<std::unique_ptr<T>> ErrorOrPointer) {
  if (ErrorOrPointer)
    return std::move(*ErrorOrPointer);
  return nullptr;
}

template <class T>
static std::unique_ptr<T>
expectedToPointer(Expected<std::unique_ptr<T>> ExpectedPointer) {
  if (ExpectedPointer)
    return std::move(*ExpectedPointer);
  consumeError(ExpectedPointer.takeError());
  return nullptr;
}

static bool
bufferHasContent(ErrorOr<std::unique_ptr<MemoryBuffer>> ErrorOrBuffer,
                 MemoryBufferRef Content) {
  if (!ErrorOrBuffer)
    return false;
  std::unique_ptr<MemoryBuffer> Buffer = std::move(*ErrorOrBuffer);
  return Buffer->getBuffer() == Content.getBuffer() &&
         Buffer->getBufferIdentifier() == Content.getBufferIdentifier();
}

static ObjectRef createBlobUnchecked(CASDB &CAS, StringRef Content) {
  return CAS.getReference(llvm::cantFail(CAS.storeFromString(None, Content)));
}

static Expected<ObjectHandle> createEmptyTree(CASDB &CAS) {
  HierarchicalTreeBuilder Builder;
  return Builder.create(CAS);
}

static Expected<ObjectHandle> createFlatTree(CASDB &CAS) {
  HierarchicalTreeBuilder Builder;
  Builder.push(createBlobUnchecked(CAS, "1"), TreeEntry::Regular, "file1");
  Builder.push(createBlobUnchecked(CAS, "1"), TreeEntry::Regular, "1");
  Builder.push(createBlobUnchecked(CAS, "2"), TreeEntry::Regular, "2");
  return Builder.create(CAS);
}

static Expected<ObjectHandle> createNestedTree(CASDB &CAS) {
  ObjectRef Data1 = createBlobUnchecked(CAS, "blob1");
  ObjectRef Data2 = createBlobUnchecked(CAS, "blob2");
  ObjectRef Data3 = createBlobUnchecked(CAS, "blob3");

  HierarchicalTreeBuilder Builder;
  Builder.push(Data2, TreeEntry::Regular, "/d2");
  Builder.push(Data1, TreeEntry::Regular, "/t1/d1");
  Builder.push(Data3, TreeEntry::Regular, "/t3/d3");
  Builder.push(Data1, TreeEntry::Regular, "/t3/t1nested/d1");
  Builder.push(Data1, TreeEntry::Regular, "/t3/t2/d1also");
  Builder.push(Data2, TreeEntry::Regular, "/t3/t2/d2");
  return Builder.create(CAS);
}

static Expected<ObjectHandle> createSymlinksTree(CASDB &CAS) {
  auto make = [&](StringRef Bytes) { return createBlobUnchecked(CAS, Bytes); };

  HierarchicalTreeBuilder Builder;
  Builder.push(make("broken"), TreeEntry::Symlink, "/s0");
  Builder.push(make("b1"), TreeEntry::Symlink, "/s1");
  Builder.push(make("blob1"), TreeEntry::Regular, "/b1");
  Builder.push(make("d/b2"), TreeEntry::Symlink, "/s2");
  Builder.push(make("blob2"), TreeEntry::Regular, "/d/b2");
  Builder.push(make("../s4"), TreeEntry::Symlink, "/d/s3");
  Builder.push(make("d/s5/b3"), TreeEntry::Symlink, "/s4");
  Builder.push(make("e/s6"), TreeEntry::Symlink, "/d/s5");
  Builder.push(make("f"), TreeEntry::Symlink, "/d/e/s6");
  Builder.push(make("blob3"), TreeEntry::Regular, "/d/e/f/b3");
  Builder.push(make("blob4"), TreeEntry::Regular, "/d/e/f/b4");
  Builder.push(make("/d/b2"), TreeEntry::Symlink, "/d/e/f/s7");
  Builder.push(make(".."), TreeEntry::Symlink, "/d/e/s8");
  return Builder.create(CAS);
}

static Expected<ObjectHandle> createSymlinkLoopsTree(CASDB &CAS) {
  auto make = [&](StringRef Bytes) { return createBlobUnchecked(CAS, Bytes); };

  HierarchicalTreeBuilder Builder;
  Builder.push(make("s0"), TreeEntry::Symlink, "/s0");
  Builder.push(make("s1"), TreeEntry::Symlink, "/s2");
  Builder.push(make("s2"), TreeEntry::Symlink, "/s1");
  Builder.push(make("../s2"), TreeEntry::Symlink, "/d/s3");
  Builder.push(make("d/s5"), TreeEntry::Symlink, "/d/s4");
  Builder.push(make("../s4"), TreeEntry::Symlink, "/d/d/s5");
  return Builder.create(CAS);
}

static Expected<std::unique_ptr<vfs::FileSystem>>
createFS(CASDB &CAS, Expected<ObjectHandle> Tree) {
  if (!Tree)
    return Tree.takeError();
  return createCASFileSystem(CAS, CAS.getID(*Tree));
}

template <class IteratorType>
static bool hasPathAndTypeBeforeIncrement(IteratorType &I, std::error_code &EC,
                                          StringRef Path,
                                          sys::fs::file_type Type) {
  if (EC)
    return false;
  bool Return = I->path() == Path && I->type() == Type;
  I.increment(EC);
  return Return;
}

template <class IteratorType>
static bool isEnd(IteratorType &I, std::error_code &EC) {
  if (EC)
    return false;
  return I == IteratorType();
}

TEST(CASFileSystemTest, getBufferForFileEmpty) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);
  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(createFS(*CAS, createEmptyTree(*CAS)).moveInto(CASFS),
                    Succeeded());
  ASSERT_FALSE(errorOrToPointer(CASFS->getBufferForFile("file")));
  ASSERT_FALSE(errorOrToPointer(CASFS->getBufferForFile("path/to/file")));
}

TEST(CASFileSystemTest, getBufferForFileFlat) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);

  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(createFS(*CAS, createFlatTree(*CAS)).moveInto(CASFS),
                    Succeeded());

  std::unique_ptr<MemoryBuffer> F1 =
      errorOrToPointer(CASFS->getBufferForFile("/1"));
  std::unique_ptr<MemoryBuffer> F2 =
      errorOrToPointer(CASFS->getBufferForFile("2"));
  std::unique_ptr<MemoryBuffer> Ffile1 =
      errorOrToPointer(CASFS->getBufferForFile("file1"));
  ASSERT_TRUE(Ffile1);
  ASSERT_TRUE(F1);
  ASSERT_TRUE(F2);
  EXPECT_EQ("/1", F1->getBufferIdentifier());
  EXPECT_EQ("2", F2->getBufferIdentifier());
  EXPECT_EQ("file1", Ffile1->getBufferIdentifier());
  EXPECT_EQ("1", F1->getBuffer());
  EXPECT_EQ("2", F2->getBuffer());
  EXPECT_EQ("1", Ffile1->getBuffer());
}

TEST(CASFileSystemTest, getBufferForFileNested) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);

  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(createFS(*CAS, createNestedTree(*CAS)).moveInto(CASFS),
                    Succeeded());

  std::unique_ptr<MemoryBuffer> T1D1 =
      errorOrToPointer(CASFS->getBufferForFile("/t1/d1"));
  std::unique_ptr<MemoryBuffer> T1NestedD1 =
      errorOrToPointer(CASFS->getBufferForFile("t3/t1nested/d1"));
  std::unique_ptr<MemoryBuffer> T3T2D1Also =
      errorOrToPointer(CASFS->getBufferForFile("/t3/t2/d1also"));
  std::unique_ptr<MemoryBuffer> T3TD3 =
      errorOrToPointer(CASFS->getBufferForFile("t3/d3"));
  ASSERT_TRUE(T1D1);
  ASSERT_TRUE(T1NestedD1);
  ASSERT_TRUE(T3T2D1Also);
  ASSERT_TRUE(T3TD3);

  EXPECT_EQ("/t1/d1", T1D1->getBufferIdentifier());
  EXPECT_EQ("t3/t1nested/d1", T1NestedD1->getBufferIdentifier());
  EXPECT_EQ("/t3/t2/d1also", T3T2D1Also->getBufferIdentifier());
  EXPECT_EQ("t3/d3", T3TD3->getBufferIdentifier());

  EXPECT_EQ("blob1", T1D1->getBuffer());
  EXPECT_EQ("blob1", T1NestedD1->getBuffer());
  EXPECT_EQ("blob1", T3T2D1Also->getBuffer());
  EXPECT_EQ("blob3", T3TD3->getBuffer());
}

TEST(CASFileSystemTest, getBufferForFileSymlinks) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);
  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(createFS(*CAS, createSymlinksTree(*CAS)).moveInto(CASFS),
                    Succeeded());

  // /s0 -> broken
  // /s1 -> b1
  // /b1
  // /s2 -> d/b2
  // /d/b2
  // /d/s3 -> ../s4
  // /s4 -> d/s5/b3
  // /d/s5 -> e/s6
  // /d/e/s6 -> f
  // /d/e/f/b4
  ErrorOr<std::unique_ptr<MemoryBuffer>> S0 = CASFS->getBufferForFile("/s0");
  EXPECT_FALSE(S0);
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/s1"),
                               MemoryBufferRef("blob1", "/s1")));
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/s2"),
                               MemoryBufferRef("blob2", "/s2")));
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/s4"),
                               MemoryBufferRef("blob3", "/s4")));
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/d/s3"),
                               MemoryBufferRef("blob3", "/d/s3")));
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/d/s5/b4"),
                               MemoryBufferRef("blob4", "/d/s5/b4")));
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/d/e/s6/b4"),
                               MemoryBufferRef("blob4", "/d/e/s6/b4")));
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/d/s5/../f/b3"),
                               MemoryBufferRef("blob3", "/d/s5/../f/b3")));
}

TEST(CASFileSystemTest, getBufferForFileSymlinkLoops) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);
  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(
      createFS(*CAS, createSymlinkLoopsTree(*CAS)).moveInto(CASFS),
      Succeeded());

  // /s0 -> s0
  // /s1 -> s2
  // /s2 -> s1
  // /d/s3 -> ../s2
  // /d/s4 -> d/s5
  // /d/d/s5 -> ../s4
  EXPECT_EQ(std::make_error_code(std::errc::too_many_symbolic_link_levels),
            CASFS->getBufferForFile("/s0").getError());
  EXPECT_EQ(std::make_error_code(std::errc::too_many_symbolic_link_levels),
            CASFS->getBufferForFile("/s1").getError());
  EXPECT_EQ(std::make_error_code(std::errc::too_many_symbolic_link_levels),
            CASFS->getBufferForFile("/s2").getError());
  EXPECT_EQ(std::make_error_code(std::errc::too_many_symbolic_link_levels),
            CASFS->getBufferForFile("/d/s3").getError());
  EXPECT_EQ(std::make_error_code(std::errc::too_many_symbolic_link_levels),
            CASFS->getBufferForFile("/d/s4").getError());
  EXPECT_EQ(std::make_error_code(std::errc::too_many_symbolic_link_levels),
            CASFS->getBufferForFile("/d/d/s5").getError());
}

TEST(CASFileSystemTest, openFileForReadEmpty) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);
  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(createFS(*CAS, createEmptyTree(*CAS)).moveInto(CASFS),
                    Succeeded());
  ASSERT_FALSE(errorOrToPointer(CASFS->openFileForRead("file")));
  ASSERT_FALSE(errorOrToPointer(CASFS->openFileForRead("path/to/file")));
}

TEST(CASFileSystemTest, openFileForReadFlat) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);

  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(createFS(*CAS, createFlatTree(*CAS)).moveInto(CASFS),
                    Succeeded());

  std::unique_ptr<vfs::File> F1 =
      errorOrToPointer(CASFS->openFileForRead("/1"));
  std::unique_ptr<vfs::File> F2 = errorOrToPointer(CASFS->openFileForRead("2"));
  std::unique_ptr<vfs::File> Ffile1 =
      errorOrToPointer(CASFS->openFileForRead("file1"));
  ASSERT_TRUE(Ffile1);
  ASSERT_TRUE(F1);
  ASSERT_TRUE(F2);
  std::unique_ptr<MemoryBuffer> B1 = errorOrToPointer(F1->getBuffer("/1"));
  std::unique_ptr<MemoryBuffer> B2 = errorOrToPointer(F2->getBuffer("2"));
  std::unique_ptr<MemoryBuffer> Bfile1 =
      errorOrToPointer(Ffile1->getBuffer("file1"));
  EXPECT_EQ("/1", B1->getBufferIdentifier());
  EXPECT_EQ("2", B2->getBufferIdentifier());
  EXPECT_EQ("file1", Bfile1->getBufferIdentifier());
  EXPECT_EQ("1", B1->getBuffer());
  EXPECT_EQ("2", B2->getBuffer());
  EXPECT_EQ("1", Bfile1->getBuffer());
}

TEST(CASFileSystemTest, getDirectoryEntry) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(createFS(*CAS, createSymlinksTree(*CAS)).moveInto(CASFS),
                    Succeeded());

  struct TestTuple {
    StringRef In;
    StringRef NoFollow;
    StringRef Follow;
  };
  TestTuple Tests[] = {
      {"/b1", "/b1", "/b1"},
      {"/s0", "/s0", ""},
      {"/s1", "/s1", "/b1"},
      {"/s2", "/s2", "/d/b2"},

      // "s8" points at a directory and is a bit more interesting.
      {"/d/e/s8", "/d/e/s8", "/d"},
      {"/d/e/s8/.", "/d", "/d"},
      {"/d/e/s8/", "/d", "/d"},
      {"/d/e/s8/e/f/s7", "/d/e/f/s7", "/d/b2"},
  };

  for (const auto &Test : makeArrayRef(Tests)) {
    const vfs::CachedDirectoryEntry *Entry = nullptr;
    ASSERT_THAT_ERROR(CASFS->getDirectoryEntry(Test.In, false).moveInto(Entry),
                      Succeeded());
    ASSERT_EQ(Test.NoFollow, Entry->getTreePath());

    Error E = CASFS->getDirectoryEntry(Test.In, true).moveInto(Entry);
    if (Test.Follow.empty()) {
      ASSERT_THAT_ERROR(std::move(E), Failed());
      continue;
    }
    ASSERT_THAT_ERROR(std::move(E), Succeeded());
    ASSERT_EQ(Test.Follow, Entry->getTreePath());
  }
}

TEST(CASFileSystemTest, getDirectoryEntrySymlinks) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);
  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(createFS(*CAS, createSymlinksTree(*CAS)).moveInto(CASFS),
                    Succeeded());

  // /s0 -> broken
  // /s1 -> b1
  // /b1
  // /s2 -> d/b2
  // /d/b2
  // /d/s3 -> ../s4
  // /s4 -> d/s5/b3
  // /d/s5 -> e/s6
  // /d/e/s6 -> f
  // /d/e/f/b4
  ErrorOr<std::unique_ptr<MemoryBuffer>> S0 = CASFS->getBufferForFile("/s0");
  EXPECT_FALSE(S0);
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/s1"),
                               MemoryBufferRef("blob1", "/s1")));
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/s2"),
                               MemoryBufferRef("blob2", "/s2")));
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/s4"),
                               MemoryBufferRef("blob3", "/s4")));
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/d/s3"),
                               MemoryBufferRef("blob3", "/d/s3")));
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/d/s5/b4"),
                               MemoryBufferRef("blob4", "/d/s5/b4")));
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/d/e/s6/b4"),
                               MemoryBufferRef("blob4", "/d/e/s6/b4")));
  EXPECT_TRUE(bufferHasContent(CASFS->getBufferForFile("/d/s5/../f/b3"),
                               MemoryBufferRef("blob3", "/d/s5/../f/b3")));
}

TEST(CASFileSystemTest, dirBeginEmpty) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);
  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(createFS(*CAS, createEmptyTree(*CAS)).moveInto(CASFS),
                    Succeeded());

  std::error_code EC;
  vfs::directory_iterator D;

  D = CASFS->dir_begin("/", EC);
  ASSERT_TRUE(isEnd(D, EC));

  D = CASFS->dir_begin(".", EC);
  ASSERT_TRUE(isEnd(D, EC));

  D = CASFS->dir_begin("././.", EC);
  ASSERT_TRUE(isEnd(D, EC));

  D = CASFS->dir_begin("dir", EC);
  EXPECT_EQ(std::make_error_code(std::errc::no_such_file_or_directory), EC);

  D = CASFS->dir_begin("/path/to/dir", EC);
  EXPECT_EQ(std::make_error_code(std::errc::no_such_file_or_directory), EC);

  D = CASFS->dir_begin("/path/..", EC);
  EXPECT_EQ(std::make_error_code(std::errc::no_such_file_or_directory), EC);
}

TEST(CASFileSystemTest, dirBeginFlat) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);

  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(createFS(*CAS, createFlatTree(*CAS)).moveInto(CASFS),
                    Succeeded());

  std::error_code EC;
  vfs::directory_iterator D;
  D = CASFS->dir_begin("/", EC);
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/1",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/2",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/file1",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(isEnd(D, EC));

  // FIXME: This seems to match other filesystems, but it seems like this
  // should be "./1"?
  D = CASFS->dir_begin(".", EC);
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/./1",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/./2",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/./file1",
                                            sys::fs::file_type::regular_file));

  D = CASFS->dir_begin("1", EC);
  EXPECT_EQ(std::make_error_code(std::errc::not_a_directory), EC);
}

TEST(CASFileSystemTest, dirBeginNested) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);

  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(createFS(*CAS, createNestedTree(*CAS)).moveInto(CASFS),
                    Succeeded());

  // Full structure:
  //
  // d2
  // t1/d1
  // t3/d3
  // t3/t1nested/d1
  // t3/t2/d1also
  // t3/t2/d2
  std::error_code EC;
  vfs::directory_iterator D;
  D = CASFS->dir_begin("/", EC);
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/d2",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(
      D, EC, "/t1", sys::fs::file_type::directory_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(
      D, EC, "/t3", sys::fs::file_type::directory_file));
  ASSERT_TRUE(isEnd(D, EC));

  // FIXME: This seems to match other filesystems, but it seems like this
  // should be "t3/d3"?
  D = CASFS->dir_begin("t3", EC);
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/t3/d3",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(
      D, EC, "/t3/t1nested", sys::fs::file_type::directory_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(
      D, EC, "/t3/t2", sys::fs::file_type::directory_file));
  ASSERT_TRUE(isEnd(D, EC));
}

TEST(CASFileSystemTest, recursiveDirectoryIteratorNested) {
  std::unique_ptr<CASDB> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);

  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(createFS(*CAS, createNestedTree(*CAS)).moveInto(CASFS),
                    Succeeded());

  // Full structure:
  //
  // d2
  // t1/d1
  // t3/d3
  // t3/t1nested/d1
  // t3/t2/d1also
  // t3/t2/d2
  std::error_code EC;
  vfs::recursive_directory_iterator D(*CASFS, "/", EC);
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/d2",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(
      D, EC, "/t1", sys::fs::file_type::directory_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/t1/d1",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(
      D, EC, "/t3", sys::fs::file_type::directory_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/t3/d3",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(
      D, EC, "/t3/t1nested", sys::fs::file_type::directory_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/t3/t1nested/d1",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(
      D, EC, "/t3/t2", sys::fs::file_type::directory_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/t3/t2/d1also",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(hasPathAndTypeBeforeIncrement(D, EC, "/t3/t2/d2",
                                            sys::fs::file_type::regular_file));
  ASSERT_TRUE(isEnd(D, EC));
}
