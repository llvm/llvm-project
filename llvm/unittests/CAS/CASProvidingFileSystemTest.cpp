//===- CASProvidingFileSystemTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/CASProvidingFileSystem.h"
#include "llvm/CAS/CASFileSystem.h"
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

  std::unique_ptr<vfs::FileSystem> VFS = createCASProvidingFileSystem(DB, FS);
  ASSERT_TRUE(VFS);
  auto &CASFS = cast<CASBackedFileSystem>(*VFS);
  {
    std::unique_ptr<CASBackedFile> File;
    ASSERT_THAT_ERROR(CASFS.openCASBackedFileForRead(Path1).moveInto(File),
                      Succeeded());
    ASSERT_TRUE(File);
    cas::ObjectRef Ref = File->getObjectRefForContent();
    std::optional<ObjectProxy> BlobContents;
    ASSERT_THAT_ERROR(DB->getProxy(Ref).moveInto(BlobContents), Succeeded());
    EXPECT_EQ(BlobContents->getData(), Contents1);
  }
  {
    std::optional<cas::ObjectRef> Ref;
    ASSERT_THAT_ERROR(CASFS.getObjectRefForFileContent(Path1).moveInto(Ref),
                      Succeeded());
    std::optional<ObjectProxy> BlobContents;
    ASSERT_THAT_ERROR(DB->getProxy(*Ref).moveInto(BlobContents), Succeeded());
    EXPECT_EQ(BlobContents->getData(), Contents1);
  }
  {
    std::optional<std::pair<std::unique_ptr<MemoryBuffer>, cas::ObjectRef>> Val;
    ASSERT_THAT_ERROR(
        CASFS
            .getBufferAndObjectRefForFile(Path2, /*FileSize*/ -1,
                                          /*RequiresNullTerminator*/ false,
                                          /*IsVolatile*/ false, /*IsText*/ true)
            .moveInto(Val),
        Succeeded());
    EXPECT_EQ(Contents2, Val->first->getBuffer());
    std::optional<ObjectProxy> BlobContents;
    ASSERT_THAT_ERROR(DB->getProxy(Val->second).moveInto(BlobContents),
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
  std::unique_ptr<vfs::FileSystem> VFS =
      createCASProvidingFileSystem(DB, std::move(UnderlyingFS));
  ASSERT_TRUE(VFS);
  auto &CASFS = cast<CASBackedFileSystem>(*VFS);

  std::unique_ptr<CASBackedFile> File;
  ASSERT_THAT_ERROR(CASFS.openCASBackedFileForRead(Path).moveInto(File),
                    Succeeded());
  cas::ObjectRef Ref = File->getObjectRefForContent();
  std::optional<ObjectProxy> BlobContents;
  ASSERT_THAT_ERROR(DB->getProxy(Ref).moveInto(BlobContents), Succeeded());
  EXPECT_EQ(BlobContents->getData(), Contents);

  CASID ID = DB->getID(Ref);
  std::optional<ObjectProxy> Proxy;
  // The file also ingested into the underlying FS.
  ASSERT_THAT_ERROR(UnderlyingDB->getProxy(ID).moveInto(Proxy), Succeeded());
}
