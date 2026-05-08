//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SandboxingFileSystem.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(SandboxingFileSystemTest, Basic) {
  auto BaseFS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
  BaseFS->addFile("//root/dir1/a1", 0, MemoryBuffer::getMemBuffer("contents"));
  BaseFS->addFile("//root/dir2/a2", 0, MemoryBuffer::getMemBuffer("contents"));
  BaseFS->addFile("//root/dir1a/aa", 0, MemoryBuffer::getMemBuffer("contents"));
  BaseFS->addFile("//root/dir3/a3", 0, MemoryBuffer::getMemBuffer("contents"));
  BaseFS->setCurrentWorkingDirectory("//root");

  std::unique_ptr<vfs::FileSystem> SandBoxFS;
  ASSERT_THAT_ERROR(
      vfs::createSandboxingFileSystem(BaseFS, {"//root/dir1", "dir3"})
          .moveInto(SandBoxFS),
      Succeeded());
  EXPECT_TRUE(SandBoxFS->exists("//root/dir1"));
  EXPECT_TRUE(SandBoxFS->exists("//root/dir1/"));
  EXPECT_TRUE(SandBoxFS->exists("//root/dir1/a1"));
  EXPECT_FALSE(SandBoxFS->exists("//root/dir2/a2"));
  EXPECT_FALSE(SandBoxFS->exists("//root/dir1a/aa"));
  EXPECT_TRUE(SandBoxFS->exists("//root/dir3/a3"));
  EXPECT_FALSE(SandBoxFS->exists("//ROOT/dir1/a1"));
  EXPECT_TRUE(SandBoxFS->exists("dir1/a1"));
  EXPECT_FALSE(SandBoxFS->exists("dir2/a2"));
}
