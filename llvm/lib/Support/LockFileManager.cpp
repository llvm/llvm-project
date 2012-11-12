//===--- LockFileManager.cpp - File-level Locking Utility------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/LockFileManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#if LLVM_ON_WIN32
#include <windows.h>
#endif
#if LLVM_ON_UNIX
#include <unistd.h>
#endif
using namespace llvm;

/// \brief Attempt to read the lock file with the given name, if it exists.
///
/// \param LockFileName The name of the lock file to read.
///
/// \returns The process ID of the process that owns this lock file
Optional<std::pair<std::string, int> >
LockFileManager::readLockFile(StringRef LockFileName) {
  // Check whether the lock file exists. If not, clearly there's nothing
  // to read, so we just return.
  bool Exists = false;
  if (sys::fs::exists(LockFileName, Exists) || !Exists)
    return Optional<std::pair<std::string, int> >();

  // Read the owning host and PID out of the lock file. If it appears that the
  // owning process is dead, the lock file is invalid.
  int PID = 0;
  std::string Hostname;
  std::ifstream Input(LockFileName.str().c_str());
  if (Input >> Hostname >> PID && PID > 0 &&
      processStillExecuting(Hostname, PID))
    return std::make_pair(Hostname, PID);

  // Delete the lock file. It's invalid anyway.
  bool Existed;
  sys::fs::remove(LockFileName, Existed);
  return Optional<std::pair<std::string, int> >();
}

bool LockFileManager::processStillExecuting(StringRef Hostname, int PID) {
#if LLVM_ON_UNIX && !defined(__ANDROID__)
  char MyHostname[256];
  MyHostname[255] = 0;
  MyHostname[0] = 0;
  gethostname(MyHostname, 255);
  // Check whether the process is dead. If so, we're done.
  if (MyHostname == Hostname && getsid(PID) == -1 && errno == ESRCH)
    return false;
#endif

  return true;
}

LockFileManager::LockFileManager(StringRef FileName)
{
  LockFileName = FileName;
  LockFileName += ".lock";

  // If the lock file already exists, don't bother to try to create our own
  // lock file; it won't work anyway. Just figure out who owns this lock file.
  if ((Owner = readLockFile(LockFileName)))
    return;

  // Create a lock file that is unique to this instance.
  UniqueLockFileName = LockFileName;
  UniqueLockFileName += "-%%%%%%%%";
  int UniqueLockFileID;
  if (error_code EC
        = sys::fs::unique_file(UniqueLockFileName.str(),
                                     UniqueLockFileID,
                                     UniqueLockFileName,
                                     /*makeAbsolute=*/false)) {
    Error = EC;
    return;
  }

  // Write our process ID to our unique lock file.
  {
    raw_fd_ostream Out(UniqueLockFileID, /*shouldClose=*/true);

#if LLVM_ON_UNIX
    // FIXME: move getpid() call into LLVM
    char hostname[256];
    hostname[255] = 0;
    hostname[0] = 0;
    gethostname(hostname, 255);
    Out << hostname << ' ' << getpid();
#else
    Out << "localhost 1";
#endif
    Out.close();

    if (Out.has_error()) {
      // We failed to write out PID, so make up an excuse, remove the
      // unique lock file, and fail.
      Error = make_error_code(errc::no_space_on_device);
      bool Existed;
      sys::fs::remove(UniqueLockFileName.c_str(), Existed);
      return;
    }
  }

  // Create a hard link from the lock file name. If this succeeds, we're done.
  error_code EC
    = sys::fs::create_hard_link(UniqueLockFileName.str(),
                                      LockFileName.str());
  if (EC == errc::success)
    return;

  // Creating the hard link failed.

#ifdef LLVM_ON_UNIX
  // The creation of the hard link may appear to fail, but if stat'ing the
  // unique file returns a link count of 2, then we can still declare success.
  struct stat StatBuf;
  if (stat(UniqueLockFileName.c_str(), &StatBuf) == 0 &&
      StatBuf.st_nlink == 2)
    return;
#endif

  // Someone else managed to create the lock file first. Wipe out our unique
  // lock file (it's useless now) and read the process ID from the lock file.
  bool Existed;
  sys::fs::remove(UniqueLockFileName.str(), Existed);
  if ((Owner = readLockFile(LockFileName)))
    return;

  // There is a lock file that nobody owns; try to clean it up and report
  // an error.
  sys::fs::remove(LockFileName.str(), Existed);
  Error = EC;
}

LockFileManager::LockFileState LockFileManager::getState() const {
  if (Owner)
    return LFS_Shared;

  if (Error)
    return LFS_Error;

  return LFS_Owned;
}

LockFileManager::~LockFileManager() {
  if (getState() != LFS_Owned)
    return;

  // Since we own the lock, remove the lock file and our own unique lock file.
  bool Existed;
  sys::fs::remove(LockFileName.str(), Existed);
  sys::fs::remove(UniqueLockFileName.str(), Existed);
}

void LockFileManager::waitForUnlock() {
  if (getState() != LFS_Shared)
    return;

#if LLVM_ON_WIN32
  unsigned long Interval = 1;
#else
  struct timespec Interval;
  Interval.tv_sec = 0;
  Interval.tv_nsec = 1000000;
#endif
  // Don't wait more than an hour for the file to appear.
  const unsigned MaxSeconds = 3600;
  do {
    // Sleep for the designated interval, to allow the owning process time to
    // finish up and remove the lock file.
    // FIXME: Should we hook in to system APIs to get a notification when the
    // lock file is deleted?
#if LLVM_ON_WIN32
    Sleep(Interval);
#else
    nanosleep(&Interval, NULL);
#endif
    // If the file no longer exists, we're done.
    bool Exists = false;
    if (!sys::fs::exists(LockFileName.str(), Exists) && !Exists)
      return;

    if (!processStillExecuting((*Owner).first, (*Owner).second))
      return;

    // Exponentially increase the time we wait for the lock to be removed.
#if LLVM_ON_WIN32
    Interval *= 2;
#else
    Interval.tv_sec *= 2;
    Interval.tv_nsec *= 2;
    if (Interval.tv_nsec >= 1000000000) {
      ++Interval.tv_sec;
      Interval.tv_nsec -= 1000000000;
    }
#endif
  } while (
#if LLVM_ON_WIN32
           Interval < MaxSeconds * 1000
#else
           Interval.tv_sec < (time_t)MaxSeconds
#endif
           );

  // Give up.
}
