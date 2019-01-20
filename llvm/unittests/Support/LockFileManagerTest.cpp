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
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;

namespace {

TEST(LockFileManagerTest, Basic) {
  SmallString<64> TmpDir;
  std::error_code EC;
  EC = sys::fs::createUniqueDirectory("LockFileManagerTestDir", TmpDir);
  ASSERT_FALSE(EC);

  SmallString<64> LockedFile(TmpDir);
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

  EC = sys::fs::remove(StringRef(TmpDir));
  ASSERT_FALSE(EC);
}

TEST(LockFileManagerTest, LinkLockExists) {
  SmallString<64> TmpDir;
  std::error_code EC;
  EC = sys::fs::createUniqueDirectory("LockFileManagerTestDir", TmpDir);
  ASSERT_FALSE(EC);

  SmallString<64> LockedFile(TmpDir);
  sys::path::append(LockedFile, "file");

  SmallString<64> FileLocK(TmpDir);
  sys::path::append(FileLocK, "file.lock");

  SmallString<64> TmpFileLock(TmpDir);
  sys::path::append(TmpFileLock, "file.lock-000");

  int FD;
  EC = sys::fs::openFileForWrite(StringRef(TmpFileLock), FD);
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

  EC = sys::fs::remove(StringRef(TmpDir));
  ASSERT_FALSE(EC);
}


TEST(LockFileManagerTest, RelativePath) {
  SmallString<64> TmpDir;
  std::error_code EC;
  EC = sys::fs::createUniqueDirectory("LockFileManagerTestDir", TmpDir);
  ASSERT_FALSE(EC);

  char PathBuf[1024];
  const char *OrigPath = getcwd(PathBuf, 1024);
  ASSERT_FALSE(chdir(TmpDir.c_str()));

  sys::fs::create_directory("inner");
  SmallString<64> LockedFile("inner");
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

  EC = sys::fs::remove("inner");
  ASSERT_FALSE(EC);

  ASSERT_FALSE(chdir(OrigPath));

  EC = sys::fs::remove(StringRef(TmpDir));
  ASSERT_FALSE(EC);
}

} // end anonymous namespace
