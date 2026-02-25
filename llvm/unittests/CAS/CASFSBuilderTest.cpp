//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASFSBuilder.h"
#include "llvm/CAS/CASFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;
using llvm::sys::fs::UniqueID;
using llvm::unittest::TempDir;
using llvm::unittest::TempFile;
using llvm::unittest::TempLink;

TEST(CASFSBuilderTest, MergeRoots) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);

  TempDir TestDirectory("casfs-builder-test", /*Unique*/ true);

  TempDir a(TestDirectory.path("a"));
  TempDir ab(TestDirectory.path("a/b"));
  TempFile f1(ab.path("f1"), "", "aaaa");
  TempDir c(TestDirectory.path("c"));
  TempDir cd(TestDirectory.path("c/d"));
  TempFile f2(cd.path("f2"), "", "aaaa");

  std::optional<ObjectProxy> aRoot;
  {
    CASFSBuilder Builder(*CAS);
    ASSERT_THAT_ERROR(Builder.ingestFileSystemPath(a.path()), Succeeded());
    ASSERT_THAT_ERROR(Builder.finish().moveInto(aRoot), Succeeded());
  }

  std::optional<ObjectProxy> cRoot;
  {
    CASFSBuilder Builder(*CAS);
    ASSERT_THAT_ERROR(Builder.ingestFileSystemPath(c.path()), Succeeded());
    ASSERT_THAT_ERROR(Builder.finish().moveInto(cRoot), Succeeded());
  }

  std::optional<ObjectProxy> mergedRoot;
  {
    CASFSBuilder Builder(*CAS);
    Builder.mergeCASFSRoot(aRoot->getRef());
    Builder.mergeCASFSRoot(cRoot->getRef());
    Builder.mergeCASFSRoot(cRoot->getRef(), "0/1");
    ASSERT_THAT_ERROR(Builder.finish().moveInto(mergedRoot), Succeeded());
  }

  std::unique_ptr<vfs::FileSystem> CASFS;
  ASSERT_THAT_ERROR(
      createCASFileSystem(*CAS, mergedRoot->getID()).moveInto(CASFS),
      Succeeded());

  std::string FSStr;
  llvm::raw_string_ostream OS(FSStr);
  CASFS->print(OS, vfs::FileSystem::PrintType::RecursiveContents);
  StringRef Printed(FSStr);
  EXPECT_TRUE(Printed.consume_front("CASFileSystem")) << Printed;
  EXPECT_TRUE(Printed.consume_front("\n  root: / llvmcas://")) << Printed;
  Printed = Printed.drop_front(64); // 32 hash bytes in hex
  EXPECT_TRUE(Printed.consume_front("\n    file llvmcas://")) << Printed;
  Printed = Printed.drop_front(65); // 32 hash bytes in hex + 1 space
  EXPECT_TRUE(Printed.consume_front("/0/1")) << Printed;
  EXPECT_TRUE(Printed.consume_front(f2.path())) << Printed;
  EXPECT_TRUE(Printed.consume_front("\n    file llvmcas://")) << Printed;
  Printed = Printed.drop_front(65); // 32 hash bytes in hex + 1 space
  EXPECT_TRUE(Printed.consume_front(f1.path())) << Printed;
  EXPECT_TRUE(Printed.consume_front("\n    file llvmcas://")) << Printed;
  Printed = Printed.drop_front(65); // 32 hash bytes in hex + 1 space
  EXPECT_TRUE(Printed.consume_front(f2.path())) << Printed;
  EXPECT_EQ(Printed, "\n");
}

TEST(CASFSBuilderTest, NoFollowSymlink) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);

  TempDir TestDirectory("casfs-builder-test", /*Unique*/ true);

  TempDir a(TestDirectory.path("a"));
  TempDir ab(TestDirectory.path("a/b"));
  TempFile f1(ab.path("f1"), "", "aaaa");
  TempLink lnk("a", TestDirectory.path("lnk"));
  TempLink abLnk1("l2", ab.path("l1"));
  TempLink abLnk2("l1", ab.path("l2"));

  std::optional<ObjectProxy> lnkRoot;
  {
    CASFSBuilder Builder(*CAS);
    ASSERT_THAT_ERROR(Builder.ingestFileSystemPath(lnk.path()), Succeeded());
    ASSERT_THAT_ERROR(Builder.finish().moveInto(lnkRoot), Succeeded());
  }

  {
    std::unique_ptr<vfs::FileSystem> CASFS;
    ASSERT_THAT_ERROR(
        createCASFileSystem(*CAS, lnkRoot->getID()).moveInto(CASFS),
        Succeeded());

    std::string FSStr;
    llvm::raw_string_ostream OS(FSStr);
    CASFS->print(OS, vfs::FileSystem::PrintType::RecursiveContents);
    StringRef Printed(FSStr);
    EXPECT_TRUE(Printed.consume_front("CASFileSystem")) << Printed;
    EXPECT_TRUE(Printed.consume_front("\n  root: / llvmcas://")) << Printed;
    Printed = Printed.drop_front(64); // 32 hash bytes in hex
    EXPECT_TRUE(Printed.consume_front("\n    syml llvmcas://")) << Printed;
    Printed = Printed.drop_front(65); // 32 hash bytes in hex + 1 space
    EXPECT_TRUE(Printed.consume_front(lnk.path())) << Printed;
    EXPECT_TRUE(Printed.consume_front(" -> a")) << Printed;
    EXPECT_EQ(Printed, "\n");
  }

  std::optional<ObjectProxy> abRoot;
  {
    CASFSBuilder Builder(*CAS);
    ASSERT_THAT_ERROR(Builder.ingestFileSystemPath(ab.path()), Succeeded());
    ASSERT_THAT_ERROR(Builder.finish().moveInto(abRoot), Succeeded());
  }

  {
    std::unique_ptr<vfs::FileSystem> CASFS;
    ASSERT_THAT_ERROR(
        createCASFileSystem(*CAS, abRoot->getID()).moveInto(CASFS),
        Succeeded());

    std::string FSStr;
    llvm::raw_string_ostream OS(FSStr);
    CASFS->print(OS, vfs::FileSystem::PrintType::RecursiveContents);
    StringRef Printed(FSStr);
    EXPECT_TRUE(Printed.consume_front("CASFileSystem")) << Printed;
    EXPECT_TRUE(Printed.consume_front("\n  root: / llvmcas://")) << Printed;
    Printed = Printed.drop_front(64); // 32 hash bytes in hex
    EXPECT_TRUE(Printed.consume_front("\n    file llvmcas://")) << Printed;
    Printed = Printed.drop_front(65); // 32 hash bytes in hex + 1 space
    EXPECT_TRUE(Printed.consume_front(f1.path())) << Printed;
    EXPECT_TRUE(Printed.consume_front("\n    syml llvmcas://")) << Printed;
    Printed = Printed.drop_front(65); // 32 hash bytes in hex + 1 space
    EXPECT_TRUE(Printed.consume_front(abLnk1.path())) << Printed;
    EXPECT_TRUE(Printed.consume_front(" -> l2")) << Printed;
    EXPECT_TRUE(Printed.consume_front("\n    syml llvmcas://")) << Printed;
    Printed = Printed.drop_front(65); // 32 hash bytes in hex + 1 space
    EXPECT_TRUE(Printed.consume_front(abLnk2.path())) << Printed;
    EXPECT_TRUE(Printed.consume_front(" -> l1")) << Printed;
    EXPECT_EQ(Printed, "\n");
  }
}

TEST(CASFSBuilderTest, Missing) {
  std::unique_ptr<ObjectStore> CAS = createInMemoryCAS();
  ASSERT_TRUE(CAS);

  TempDir TestDirectory("casfs-builder-test", /*Unique*/ true);

  std::optional<ObjectProxy> Root;
  {
    CASFSBuilder Builder(*CAS);
    EXPECT_THAT_ERROR(Builder.ingestFileSystemPath(TestDirectory.path("a")),
                      Failed());
  }
}
