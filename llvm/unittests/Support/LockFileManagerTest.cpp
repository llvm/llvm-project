//===- unittests/LockFileManagerTest.cpp - LockFileManager tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;
using llvm::unittest::TempDir;

namespace {

TEST(LockFileManagerTest, Basic) {
  TempDir TmpDir("LockFileManagerTestDir", /*Unique*/ true);

  SmallString<64> LockedFile(TmpDir.path());
  sys::path::append(LockedFile, "file.lock");

  {
    // The lock file should not exist, so we should successfully acquire it.
    LockFileManager Locked1(LockedFile);
    EXPECT_EQ(LockFileManager::LFS_Owned, Locked1.getState());

    // Attempting to reacquire the lock should fail.  Waiting on it would cause
    // deadlock, so don't try that.
    LockFileManager Locked2(LockedFile);
    EXPECT_NE(LockFileManager::LFS_Owned, Locked2.getState());
  }

  // Now that the lock is out of scope, the file should be gone.
  EXPECT_FALSE(sys::fs::exists(StringRef(LockedFile)));
}

TEST(LockFileManagerTest, LinkLockExists) {
  TempDir LockFileManagerTestDir("LockFileManagerTestDir", /*Unique*/ true);

  SmallString<64> LockedFile(LockFileManagerTestDir.path());
  sys::path::append(LockedFile, "file");

  SmallString<64> FileLocK(LockFileManagerTestDir.path());
  sys::path::append(FileLocK, "file.lock");

  SmallString<64> TmpFileLock(LockFileManagerTestDir.path());
  sys::path::append(TmpFileLock, "file.lock-000");

  int FD;
  std::error_code EC = sys::fs::openFileForWrite(StringRef(TmpFileLock), FD);
  ASSERT_FALSE(EC);

  int Ret = close(FD);
  ASSERT_EQ(Ret, 0);

  EC = sys::fs::create_link(TmpFileLock.str(), FileLocK.str());
  ASSERT_FALSE(EC);

  EC = sys::fs::remove(StringRef(TmpFileLock));
  ASSERT_FALSE(EC);

  {
    // The lock file doesn't point to a real file, so we should successfully
    // acquire it.
    LockFileManager Locked(LockedFile);
    EXPECT_EQ(LockFileManager::LFS_Owned, Locked.getState());
  }

  // Now that the lock is out of scope, the file should be gone.
  EXPECT_FALSE(sys::fs::exists(StringRef(LockedFile)));
}


TEST(LockFileManagerTest, RelativePath) {
  TempDir LockFileManagerTestDir("LockFileManagerTestDir", /*Unique*/ true);

  char PathBuf[1024];
  const char *OrigPath = getcwd(PathBuf, 1024);
  ASSERT_FALSE(chdir(LockFileManagerTestDir.c_str()));

  TempDir inner("inner");
  SmallString<64> LockedFile(inner.path());
  sys::path::append(LockedFile, "file");

  SmallString<64> FileLock(LockedFile);
  FileLock += ".lock";

  {
    // The lock file should not exist, so we should successfully acquire it.
    LockFileManager Locked(LockedFile);
    EXPECT_EQ(LockFileManager::LFS_Owned, Locked.getState());
    EXPECT_TRUE(sys::fs::exists(FileLock.str()));
  }

  // Now that the lock is out of scope, the file should be gone.
  EXPECT_FALSE(sys::fs::exists(LockedFile.str()));
  EXPECT_FALSE(sys::fs::exists(FileLock.str()));

  ASSERT_FALSE(chdir(OrigPath));
}

} // end anonymous namespace
