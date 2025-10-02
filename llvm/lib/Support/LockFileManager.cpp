//===--- LockFileManager.cpp - File-level Locking Utility------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/LockFileManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/ExponentialBackoff.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <cerrno>
#include <chrono>
#include <ctime>
#include <memory>
#include <system_error>
#include <tuple>

#ifdef _WIN32
#include <windows.h>
#endif
#if LLVM_ON_UNIX
#include <unistd.h>
#endif

#if defined(__APPLE__) && defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ > 1050)
#define USE_OSX_GETHOSTUUID 1
#else
#define USE_OSX_GETHOSTUUID 0
#endif

#if USE_OSX_GETHOSTUUID
#include <uuid/uuid.h>
#endif

using namespace llvm;

/// Attempt to read the lock file with the given name, if it exists.
///
/// \param LockFileName The name of the lock file to read.
///
/// \returns The process ID of the process that owns this lock file
std::optional<LockFileManager::OwnedByAnother>
LockFileManager::readLockFile(StringRef LockFileName) {
  // Read the owning host and PID out of the lock file. If it appears that the
  // owning process is dead, the lock file is invalid.
  ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
      MemoryBuffer::getFile(LockFileName);
  if (!MBOrErr) {
    sys::fs::remove(LockFileName);
    return std::nullopt;
  }
  MemoryBuffer &MB = *MBOrErr.get();

  StringRef Hostname;
  StringRef PIDStr;
  std::tie(Hostname, PIDStr) = getToken(MB.getBuffer(), " ");
  PIDStr = PIDStr.substr(PIDStr.find_first_not_of(' '));
  int PID;
  if (!PIDStr.getAsInteger(10, PID)) {
    OwnedByAnother Owner;
    Owner.OwnerHostName = Hostname;
    Owner.OwnerPID = PID;
    if (processStillExecuting(Owner.OwnerHostName, Owner.OwnerPID))
      return Owner;
  }

  // Delete the lock file. It's invalid anyway.
  sys::fs::remove(LockFileName);
  return std::nullopt;
}

static std::error_code getHostID(SmallVectorImpl<char> &HostID) {
  HostID.clear();

#if USE_OSX_GETHOSTUUID
  // On OS X, use the more stable hardware UUID instead of hostname.
  struct timespec wait = {1, 0}; // 1 second.
  uuid_t uuid;
  if (gethostuuid(uuid, &wait) != 0)
    return errnoAsErrorCode();

  uuid_string_t UUIDStr;
  uuid_unparse(uuid, UUIDStr);
  StringRef UUIDRef(UUIDStr);
  HostID.append(UUIDRef.begin(), UUIDRef.end());

#elif LLVM_ON_UNIX
  char HostName[256];
  HostName[255] = 0;
  HostName[0] = 0;
  gethostname(HostName, 255);
  StringRef HostNameRef(HostName);
  HostID.append(HostNameRef.begin(), HostNameRef.end());

#else
  StringRef Dummy("localhost");
  HostID.append(Dummy.begin(), Dummy.end());
#endif

  return std::error_code();
}

bool LockFileManager::processStillExecuting(StringRef HostID, int PID) {
#if LLVM_ON_UNIX && !defined(__ANDROID__)
  SmallString<256> StoredHostID;
  if (getHostID(StoredHostID))
    return true; // Conservatively assume it's executing on error.

  // Check whether the process is dead. If so, we're done.
  if (StoredHostID == HostID && getsid(PID) == -1 && errno == ESRCH)
    return false;
#endif

  return true;
}

namespace {

/// An RAII helper object ensure that the unique lock file is removed.
///
/// Ensures that if there is an error or a signal before we finish acquiring the
/// lock, the unique file will be removed. And if we successfully take the lock,
/// the signal handler is left in place so that signals while the lock is held
/// will remove the unique lock file. The caller should ensure there is a
/// matching call to sys::DontRemoveFileOnSignal when the lock is released.
class RemoveUniqueLockFileOnSignal {
  StringRef Filename;
  bool RemoveImmediately;
public:
  RemoveUniqueLockFileOnSignal(StringRef Name)
  : Filename(Name), RemoveImmediately(true) {
    sys::RemoveFileOnSignal(Filename, nullptr);
  }

  ~RemoveUniqueLockFileOnSignal() {
    if (!RemoveImmediately) {
      // Leave the signal handler enabled. It will be removed when the lock is
      // released.
      return;
    }
    sys::fs::remove(Filename);
    sys::DontRemoveFileOnSignal(Filename);
  }

  void lockAcquired() { RemoveImmediately = false; }
};

} // end anonymous namespace

LockFileManager::LockFileManager(StringRef FileName)
    : FileName(FileName), Owner(OwnerUnknown{}) {}

Expected<bool> LockFileManager::tryLock() {
  assert(std::holds_alternative<OwnerUnknown>(Owner) &&
         "lock has already been attempted");

  SmallString<128> AbsoluteFileName(FileName);
  if (std::error_code EC = sys::fs::make_absolute(AbsoluteFileName))
    return createStringError(EC, "failed to obtain absolute path for " +
                                     AbsoluteFileName);
  LockFileName = AbsoluteFileName;
  LockFileName += ".lock";

  // If the lock file already exists, don't bother to try to create our own
  // lock file; it won't work anyway. Just figure out who owns this lock file.
  if (auto LockFileOwner = readLockFile(LockFileName)) {
    Owner = std::move(*LockFileOwner);
    return false;
  }

  // Create a lock file that is unique to this instance.
  UniqueLockFileName = LockFileName;
  UniqueLockFileName += "-%%%%%%%%";
  int UniqueLockFileID;
  if (std::error_code EC = sys::fs::createUniqueFile(
          UniqueLockFileName, UniqueLockFileID, UniqueLockFileName))
    return createStringError(EC, "failed to create unique file " +
                                     UniqueLockFileName);

  // Clean up the unique file on signal or scope exit.
  RemoveUniqueLockFileOnSignal RemoveUniqueFile(UniqueLockFileName);

  // Write our process ID to our unique lock file.
  {
    SmallString<256> HostID;
    if (auto EC = getHostID(HostID))
      return createStringError(EC, "failed to get host id");

    raw_fd_ostream Out(UniqueLockFileID, /*shouldClose=*/true);
    Out << HostID << ' ' << sys::Process::getProcessId();
    Out.close();

    if (Out.has_error()) {
      // We failed to write out PID, so report the error and fail.
      Error Err = createStringError(Out.error(),
                                    "failed to write to " + UniqueLockFileName);
      // Don't call report_fatal_error.
      Out.clear_error();
      return std::move(Err);
    }
  }

  while (true) {
    // Create a link from the lock file name. If this succeeds, we're done.
    std::error_code EC =
        sys::fs::create_link(UniqueLockFileName, LockFileName);
    if (!EC) {
      RemoveUniqueFile.lockAcquired();
      Owner = OwnedByUs{};
      return true;
    }

    if (EC != errc::file_exists)
      return createStringError(EC, "failed to create link " + LockFileName +
                                       " to " + UniqueLockFileName);

    // Someone else managed to create the lock file first. Read the process ID
    // from the lock file.
    if (auto LockFileOwner = readLockFile(LockFileName)) {
      Owner = std::move(*LockFileOwner);
      return false;
    }

    if (!sys::fs::exists(LockFileName)) {
      // The previous owner released the lock file before we could read it.
      // Try to get ownership again.
      continue;
    }

    // There is a lock file that nobody owns; try to clean it up and get
    // ownership.
    if ((EC = sys::fs::remove(LockFileName)))
      return createStringError(EC, "failed to remove lockfile " +
                                       UniqueLockFileName);
  }
}

LockFileManager::~LockFileManager() {
  if (!std::holds_alternative<OwnedByUs>(Owner))
    return;

  // Since we own the lock, remove the lock file and our own unique lock file.
  sys::fs::remove(LockFileName);
  sys::fs::remove(UniqueLockFileName);
  // The unique file is now gone, so remove it from the signal handler. This
  // matches a sys::RemoveFileOnSignal() in LockFileManager().
  sys::DontRemoveFileOnSignal(UniqueLockFileName);
}

WaitForUnlockResult
LockFileManager::waitForUnlockFor(std::chrono::seconds MaxSeconds) {
  auto *LockFileOwner = std::get_if<OwnedByAnother>(&Owner);
  assert(LockFileOwner &&
         "waiting for lock to be unlocked without knowing the owner");

  // Since we don't yet have an event-based method to wait for the lock file,
  // use randomized exponential backoff, similar to Ethernet collision
  // algorithm. This improves performance on machines with high core counts
  // when the file lock is heavily contended by multiple clang processes
  using namespace std::chrono_literals;
  ExponentialBackoff Backoff(MaxSeconds, 10ms, 500ms);

  // Wait first as this is only called when the lock is known to be held.
  while (Backoff.waitForNextAttempt()) {
    // FIXME: implement event-based waiting
    if (sys::fs::access(LockFileName.c_str(), sys::fs::AccessMode::Exist) ==
        errc::no_such_file_or_directory)
      return WaitForUnlockResult::Success;

    // If the process owning the lock died without cleaning up, just bail out.
    if (!processStillExecuting(LockFileOwner->OwnerHostName,
                               LockFileOwner->OwnerPID))
      return WaitForUnlockResult::OwnerDied;
  }

  // Give up.
  return WaitForUnlockResult::Timeout;
}

std::error_code LockFileManager::unsafeMaybeUnlock() {
  return sys::fs::remove(LockFileName);
}
