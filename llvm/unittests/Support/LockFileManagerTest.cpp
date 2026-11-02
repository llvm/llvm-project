//===- unittests/LockFileManagerTest.cpp - LockFileManager tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;
using llvm::unittest::TempDir;

// From TestMain.cpp.
extern const char *TestMainArgv0;

namespace {

TEST(LockFileManagerTest, Basic) {
  TempDir TmpDir("LockFileManagerTestDir", /*Unique*/ true);

  SmallString<64> LockedFile(TmpDir.path());
  sys::path::append(LockedFile, "file.lock");

  {
    // The lock file should not exist, so we should successfully acquire it.
    LockFileManager Locked1(LockedFile);
    EXPECT_THAT_EXPECTED(Locked1.tryLock(), HasValue(true));

    // Attempting to reacquire the lock should fail.  Waiting on it would cause
    // deadlock, so don't try that.
    LockFileManager Locked2(LockedFile);
    EXPECT_THAT_EXPECTED(Locked2.tryLock(), HasValue(false));
  }

  // Now that the lock is out of scope, the file should be gone.
  EXPECT_FALSE(sys::fs::exists(LockedFile.str()));
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
  std::error_code EC = sys::fs::openFileForWrite(TmpFileLock.str(), FD);
  ASSERT_FALSE(EC);

  int Ret = close(FD);
  ASSERT_EQ(Ret, 0);

  EC = sys::fs::create_link(TmpFileLock.str(), FileLocK.str());
  ASSERT_FALSE(EC);

  EC = sys::fs::remove(TmpFileLock.str());
  ASSERT_FALSE(EC);

  {
    // The lock file doesn't point to a real file, so we should successfully
    // acquire it.
    LockFileManager Locked(LockedFile);
    EXPECT_THAT_EXPECTED(Locked.tryLock(), HasValue(true));
  }

  // Now that the lock is out of scope, the file should be gone.
  EXPECT_FALSE(sys::fs::exists(LockedFile.str()));
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
    EXPECT_THAT_EXPECTED(Locked.tryLock(), HasValue(true));
    EXPECT_TRUE(sys::fs::exists(FileLock.str()));
  }

  // Now that the lock is out of scope, the file should be gone.
  EXPECT_FALSE(sys::fs::exists(LockedFile.str()));
  EXPECT_FALSE(sys::fs::exists(FileLock.str()));

  ASSERT_FALSE(chdir(OrigPath));
}

TEST(LockFileManagerTest, NonExistentDirectory) {
  TempDir LockFileManagerTestDir("LockFileManagerTestDir", /*Unique*/ true);

  SmallString<64> NonExistentDir(LockFileManagerTestDir.path());
  sys::path::append(NonExistentDir, "does_not_exist");

  SmallString<64> LockedFile(NonExistentDir);
  sys::path::append(LockedFile, "file");

  SmallString<64> FileLock(LockedFile);
  FileLock += ".lock";

  // Ensure directory truly does not exist before test
  EXPECT_FALSE(sys::fs::exists(NonExistentDir));

  {
    LockFileManager Locked(LockedFile);
    EXPECT_THAT_EXPECTED(Locked.tryLock(), HasValue(true));

    // Directory and lock file should now exist
    EXPECT_TRUE(sys::fs::exists(NonExistentDir));
    EXPECT_TRUE(sys::fs::exists(FileLock));
  }

  // After destruction, lock file should be removed
  EXPECT_FALSE(sys::fs::exists(FileLock));
}

static int LockFileManagerTestAnchor;

TEST(LockFileManagerTest, LockOwnedByExitedProcessIsReclaimed) {
  TempDir LockFileManagerTestDir("LockFileManagerTestDir", /*Unique*/ true);

  SmallString<64> LockedFile(LockFileManagerTestDir.path());
  sys::path::append(LockedFile, "file");

  SmallString<64> FileLock(LockedFile);
  FileLock += ".lock";

  // Acquire and release the lock once to find out how this machine identifies
  // itself in lock files.
  std::string HostID;
  {
    LockFileManager Locked(LockedFile);
    ASSERT_THAT_EXPECTED(Locked.tryLock(), HasValue(true));
    ErrorOr<std::unique_ptr<MemoryBuffer>> Contents =
        MemoryBuffer::getFile(FileLock);
    ASSERT_TRUE(bool(Contents));
    HostID = (*Contents)->getBuffer().split(' ').first.str();
  }
  ASSERT_FALSE(sys::fs::exists(FileLock));

  // Run a short-lived child process to get the ID of a process that exited.
  std::string Executable =
      sys::fs::getMainExecutable(TestMainArgv0, &LockFileManagerTestAnchor);
  StringRef Argv[] = {Executable, "--gtest_list_tests"};
  std::optional<StringRef> Redirects[] = {StringRef(""), StringRef(""),
                                          StringRef("")};
  std::string Error;
  sys::ProcessInfo Child =
      sys::ExecuteNoWait(Executable, Argv, /*Env=*/std::nullopt, Redirects,
                         /*MemoryLimit=*/0, &Error);
  ASSERT_NE(Child.Pid, sys::ProcessInfo::InvalidPid) << Error;
  sys::ProcessInfo Result = sys::Wait(Child, /*SecondsToWait=*/std::nullopt);
  ASSERT_EQ(Result.ReturnCode, 0);

  // Leave behind a lock file owned by that process, as if it had been
  // terminated while holding the lock.
  {
    std::error_code EC;
    raw_fd_ostream Out(FileLock, EC);
    ASSERT_FALSE(EC);
    Out << HostID << ' ' << Child.Pid;
  }

  {
    // The owner of the lock file has exited, so we should successfully acquire
    // the lock.
    LockFileManager Locked(LockedFile);
    EXPECT_THAT_EXPECTED(Locked.tryLock(), HasValue(true));
  }

  // Now that the lock is out of scope, the file should be gone.
  EXPECT_FALSE(sys::fs::exists(FileLock));
}

} // end anonymous namespace
