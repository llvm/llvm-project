//===- OnDiskCommon.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OnDiskCommon.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Process.h"
#include <mutex>
#include <thread>

#if __has_include(<sys/file.h>)
#include <sys/file.h>
#ifdef LOCK_SH
#define HAVE_FLOCK 1
#else
#define HAVE_FLOCK 0
#endif
#endif

#if __has_include(<fcntl.h>)
#include <fcntl.h>
#endif

#if __has_include(<sys/mount.h>)
#include <sys/mount.h> // statfs
#endif

using namespace llvm;

static uint64_t OnDiskCASMaxMappingSize = 0;

Expected<std::optional<uint64_t>> cas::ondisk::getOverriddenMaxMappingSize() {
  static std::once_flag Flag;
  Error Err = Error::success();
  std::call_once(Flag, [&Err] {
    ErrorAsOutParameter EAO(&Err);
    constexpr const char *EnvVar = "LLVM_CAS_MAX_MAPPING_SIZE";
    auto Value = sys::Process::GetEnv(EnvVar);
    if (!Value)
      return;

    uint64_t Size;
    if (StringRef(*Value).getAsInteger(/*auto*/ 0, Size))
      Err = createStringError(inconvertibleErrorCode(),
                              "invalid value for %s: expected integer", EnvVar);
    OnDiskCASMaxMappingSize = Size;
  });

  if (Err)
    return std::move(Err);

  if (OnDiskCASMaxMappingSize == 0)
    return std::nullopt;

  return OnDiskCASMaxMappingSize;
}

void cas::ondisk::setMaxMappingSize(uint64_t Size) {
  OnDiskCASMaxMappingSize = Size;
}

std::error_code cas::ondisk::lockFileThreadSafe(int FD,
                                                sys::fs::LockKind Kind) {
#if HAVE_FLOCK
  if (flock(FD, Kind == sys::fs::LockKind::Exclusive ? LOCK_EX : LOCK_SH) == 0)
    return std::error_code();
  return std::error_code(errno, std::generic_category());
#elif defined(_WIN32)
  // On Windows this implementation is thread-safe.
  return sys::fs::lockFile(FD, Kind);
#else
  return make_error_code(std::errc::no_lock_available);
#endif
}

std::error_code cas::ondisk::unlockFileThreadSafe(int FD) {
#if HAVE_FLOCK
  if (flock(FD, LOCK_UN) == 0)
    return std::error_code();
  return std::error_code(errno, std::generic_category());
#elif defined(_WIN32)
  // On Windows this implementation is thread-safe.
  return sys::fs::unlockFile(FD);
#else
  return make_error_code(std::errc::no_lock_available);
#endif
}

std::error_code
cas::ondisk::tryLockFileThreadSafe(int FD, std::chrono::milliseconds Timeout,
                                   sys::fs::LockKind Kind) {
#if HAVE_FLOCK
  auto Start = std::chrono::steady_clock::now();
  auto End = Start + Timeout;
  do {
    if (flock(FD, (Kind == sys::fs::LockKind::Exclusive ? LOCK_EX : LOCK_SH) |
                      LOCK_NB) == 0)
      return std::error_code();
    int Error = errno;
    if (Error == EWOULDBLOCK) {
      // Match sys::fs::tryLockFile, which sleeps for 1 ms per attempt.
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    return std::error_code(Error, std::generic_category());
  } while (std::chrono::steady_clock::now() < End);
  return make_error_code(std::errc::no_lock_available);
#elif defined(_WIN32)
  // On Windows this implementation is thread-safe.
  return sys::fs::tryLockFile(FD, Timeout, Kind);
#else
  return make_error_code(std::errc::no_lock_available);
#endif
}

Expected<size_t> cas::ondisk::preallocateFileTail(int FD, size_t CurrentSize,
                                                  size_t NewSize) {
  auto CreateError = [&](std::error_code EC) -> Expected<size_t> {
    if (EC == std::errc::not_supported)
      // Ignore ENOTSUP in case the filesystem cannot preallocate.
      return NewSize;
#if defined(HAVE_POSIX_FALLOCATE)
    if (EC == std::errc::invalid_argument && CurrentSize < NewSize && // len > 0
        NewSize < std::numeric_limits<off_t>::max()) // 0 <= offset, len < max
      // Prior to 2024, POSIX required EINVAL for cases that should be ENOTSUP,
      // so handle it the same as above if it is not one of the other ways to
      // get EINVAL.
      return NewSize;
#endif
    return createStringError(EC,
                             "failed to allocate to CAS file: " + EC.message());
  };
#if defined(HAVE_POSIX_FALLOCATE)
  // Note: posix_fallocate returns its error directly, not via errno.
  if (int Err = posix_fallocate(FD, CurrentSize, NewSize - CurrentSize))
    return CreateError(std::error_code(Err, std::generic_category()));
  return NewSize;
#elif defined(__APPLE__)
  fstore_t FAlloc;
  FAlloc.fst_flags = F_ALLOCATEALL;
#if defined(F_ALLOCATEPERSIST) &&                                              \
    defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) &&                  \
    __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ >= 130000
  // F_ALLOCATEPERSIST is introduced in macOS 13.
  FAlloc.fst_flags |= F_ALLOCATEPERSIST;
#endif
  FAlloc.fst_posmode = F_PEOFPOSMODE;
  FAlloc.fst_offset = 0;
  FAlloc.fst_length = NewSize - CurrentSize;
  FAlloc.fst_bytesalloc = 0;
  if (fcntl(FD, F_PREALLOCATE, &FAlloc))
    return CreateError(errnoAsErrorCode());
  assert(CurrentSize + FAlloc.fst_bytesalloc >= NewSize);
  return CurrentSize + FAlloc.fst_bytesalloc;
#else
  (void)CreateError; // Silence unused variable.
  return NewSize;    // Pretend it worked.
#endif
}

bool cas::ondisk::useSmallMappingSize(const Twine &P) {
  // Add exceptions to use small database file here.
#if defined(__APPLE__) && __has_include(<sys/mount.h>)
  // macOS tmpfs does not support sparse tails.
  SmallString<128> PathStorage;
  StringRef Path = P.toNullTerminatedStringRef(PathStorage);
  struct statfs StatFS;
  if (statfs(Path.data(), &StatFS) != 0)
    return false;

  if (strcmp(StatFS.f_fstypename, "tmpfs") == 0)
    return true;
#endif
  // Default to use regular database file.
  return false;
}
