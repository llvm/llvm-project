//===- CASProvidingFileSystemTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASProvidingFileSystem.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::cas;

TEST(CASProvidingFileSystemTest, Basic) {
  std::shared_ptr<ObjectStore> DB = createInMemoryCAS();
  auto FS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
  StringRef Path1 = "a.txt";
  StringRef Contents1 = "a";
  StringRef Path2 = "b.txt";
  StringRef Contents2 = "b";
  FS->addFile(Path1, 0, MemoryBuffer::getMemBuffer(Contents1));
  FS->addFile(Path2, 0, MemoryBuffer::getMemBuffer(Contents2));

  std::unique_ptr<vfs::FileSystem> CASFS = createCASProvidingFileSystem(DB, FS);
  ASSERT_TRUE(CASFS);
  {
    ErrorOr<std::unique_ptr<vfs::File>> File = CASFS->openFileForRead(Path1);
    ASSERT_TRUE(File);
    ASSERT_TRUE(*File);
    ErrorOr<std::optional<cas::ObjectRef>> Ref =
        (*File)->getObjectRefForContent();
    ASSERT_TRUE(Ref);
    ASSERT_TRUE(*Ref);
    std::optional<ObjectProxy> BlobContents;
    ASSERT_THAT_ERROR(DB->getProxy(**Ref).moveInto(BlobContents), Succeeded());
    EXPECT_EQ(BlobContents->getData(), Contents1);
  }
  {
    ErrorOr<std::optional<cas::ObjectRef>> Ref =
        CASFS->getObjectRefForFileContent(Path1);
    ASSERT_TRUE(Ref);
    ASSERT_TRUE(*Ref);
    std::optional<ObjectProxy> BlobContents;
    ASSERT_THAT_ERROR(DB->getProxy(**Ref).moveInto(BlobContents), Succeeded());
    EXPECT_EQ(BlobContents->getData(), Contents1);
  }
  {
    std::optional<cas::ObjectRef> CASContents;
    auto Buf = CASFS->getBufferForFile(Path2, /*FileSize*/ -1,
                                       /*RequiresNullTerminator*/ false,
                                       /*IsVolatile*/ false, &CASContents);
    ASSERT_TRUE(Buf);
    EXPECT_EQ(Contents2, (*Buf)->getBuffer());
    ASSERT_TRUE(CASContents);
    std::optional<ObjectProxy> BlobContents;
    ASSERT_THAT_ERROR(DB->getProxy(*CASContents).moveInto(BlobContents),
                      Succeeded());
    EXPECT_EQ(BlobContents->getData(), Contents2);
  }
}

TEST(CASProvidingFileSystemTest, WithCASSupportingFS) {
  std::shared_ptr<ObjectStore> UnderlyingDB = createInMemoryCAS();
  auto FS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
  StringRef Path = "a.txt";
  StringRef Contents = "a";
  FS->addFile(Path, 0, MemoryBuffer::getMemBuffer(Contents));
  std::unique_ptr<vfs::FileSystem> UnderlyingFS =
      createCASProvidingFileSystem(UnderlyingDB, FS);
  ASSERT_TRUE(UnderlyingFS);

  std::shared_ptr<ObjectStore> DB = createInMemoryCAS();
  std::unique_ptr<vfs::FileSystem> CASFS =
      createCASProvidingFileSystem(DB, std::move(UnderlyingFS));
  ASSERT_TRUE(CASFS);

  ErrorOr<std::unique_ptr<vfs::File>> File = CASFS->openFileForRead(Path);
  ASSERT_TRUE(File);
  ASSERT_TRUE(*File);
  ErrorOr<std::optional<cas::ObjectRef>> Ref =
      (*File)->getObjectRefForContent();
  ASSERT_TRUE(Ref);
  ASSERT_TRUE(*Ref);
  std::optional<ObjectProxy> BlobContents;
  ASSERT_THAT_ERROR(UnderlyingDB->getProxy(**Ref).moveInto(BlobContents),
                    Succeeded());
  EXPECT_EQ(BlobContents->getData(), Contents);

  CASID ID = UnderlyingDB->getID(**Ref);
  std::optional<ObjectProxy> Proxy;
  // It didn't have to ingest in DB because the underlying FS provided a CAS
  // reference.
  ASSERT_THAT_ERROR(DB->getProxy(ID).moveInto(Proxy), Failed());
}
