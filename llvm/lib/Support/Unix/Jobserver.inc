//===- llvm/Support/Unix/Jobserver.inc - Unix Jobserver Impl ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the UNIX-specific parts of the JobserverClient class.
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cassert>
#include <cerrno>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
/// Returns true if the given file descriptor is a FIFO (named pipe).
bool isFifo(int FD) {
  struct stat StatBuf;
  if (::fstat(FD, &StatBuf) != 0)
    return false;
  return S_ISFIFO(StatBuf.st_mode);
}

/// Returns true if the given file descriptors are valid.
bool areFdsValid(int ReadFD, int WriteFD) {
  if (ReadFD == -1 || WriteFD == -1)
    return false;
  // Check if the file descriptors are actually valid by checking their flags.
  return ::fcntl(ReadFD, F_GETFD) != -1 && ::fcntl(WriteFD, F_GETFD) != -1;
}
} // namespace

/// The constructor sets up the client based on the provided configuration.
/// For pipe-based jobservers, it duplicates the inherited file descriptors,
/// sets them to close-on-exec, and makes the read descriptor non-blocking.
/// For FIFO-based jobservers, it opens the named pipe. After setup, it drains
/// all available tokens from the jobserver to determine the total number of
/// available jobs (`NumJobs`), then immediately releases them back.
JobserverClientImpl::JobserverClientImpl(const JobserverConfig &Config) {
  switch (Config.TheMode) {
  case JobserverConfig::PosixPipe: {
    // Duplicate the read and write file descriptors.
    int NewReadFD = ::dup(Config.PipeFDs.Read);
    if (NewReadFD < 0)
      return;
    int NewWriteFD = ::dup(Config.PipeFDs.Write);
    if (NewWriteFD < 0) {
      ::close(NewReadFD);
      return;
    }
    // Set the new descriptors to be closed automatically on exec().
    if (::fcntl(NewReadFD, F_SETFD, FD_CLOEXEC) == -1 ||
        ::fcntl(NewWriteFD, F_SETFD, FD_CLOEXEC) == -1) {
      ::close(NewReadFD);
      ::close(NewWriteFD);
      return;
    }
    // Set the read descriptor to non-blocking.
    int flags = ::fcntl(NewReadFD, F_GETFL, 0);
    if (flags == -1 || ::fcntl(NewReadFD, F_SETFL, flags | O_NONBLOCK) == -1) {
      ::close(NewReadFD);
      ::close(NewWriteFD);
      return;
    }
    ReadFD = NewReadFD;
    WriteFD = NewWriteFD;
    break;
  }
  case JobserverConfig::PosixFifo:
    // Open the FIFO for reading. It must be non-blocking and close-on-exec.
    ReadFD = ::open(Config.Path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (ReadFD < 0 || !isFifo(ReadFD)) {
      if (ReadFD >= 0)
        ::close(ReadFD);
      ReadFD = -1;
      return;
    }
    FifoPath = Config.Path;
    // The write FD is opened on-demand in release().
    WriteFD = -1;
    break;
  default:
    return;
  }

  IsInitialized = true;
  // Determine the total number of jobs by acquiring all available slots and
  // then immediately releasing them.
  SmallVector<JobSlot, 8> Slots;
  while (true) {
    auto S = tryAcquire();
    if (!S.isValid())
      break;
    Slots.push_back(std::move(S));
  }
  NumJobs = Slots.size();
  assert(NumJobs >= 1 && "Invalid number of jobs");
  for (auto &S : Slots)
    release(std::move(S));
}

/// The destructor closes any open file descriptors.
JobserverClientImpl::~JobserverClientImpl() {
  if (ReadFD >= 0)
    ::close(ReadFD);
  if (WriteFD >= 0)
    ::close(WriteFD);
}

/// Tries to acquire a job slot. The first call to this function will always
/// successfully acquire the single "implicit" slot that is granted to every
/// process started by `make`. Subsequent calls attempt to read a one-byte
/// token from the jobserver's read pipe. A successful read grants one
/// explicit job slot. The read is non-blocking; if no token is available,
/// it fails and returns an invalid JobSlot.
JobSlot JobserverClientImpl::tryAcquire() {
  if (!IsInitialized)
    return JobSlot();

  // The first acquisition is always for the implicit slot.
  if (HasImplicitSlot.exchange(false, std::memory_order_acquire)) {
    LLVM_DEBUG(dbgs() << "Acquired implicit job slot.\n");
    return JobSlot::createImplicit();
  }

  char Token;
  ssize_t Ret;
  LLVM_DEBUG(dbgs() << "Attempting to read token from FD " << ReadFD << ".\n");
  // Loop to retry on EINTR (interrupted system call).
  do {
    Ret = ::read(ReadFD, &Token, 1);
  } while (Ret < 0 && errno == EINTR);

  if (Ret == 1) {
    LLVM_DEBUG(dbgs() << "Acquired explicit token '" << Token << "'.\n");
    return JobSlot::createExplicit(static_cast<uint8_t>(Token));
  }

  LLVM_DEBUG(dbgs() << "Failed to acquire job slot, read returned " << Ret
                    << ".\n");
  return JobSlot();
}

/// Releases a job slot back to the pool. If the slot is implicit, it simply
/// resets a flag. If the slot is explicit, it writes the character token
/// associated with the slot back into the jobserver's write pipe. For FIFO
/// jobservers, this may require opening the FIFO for writing if it hasn't
/// been already.
void JobserverClientImpl::release(JobSlot Slot) {
  if (!Slot.isValid())
    return;

  // Releasing the implicit slot just makes it available for the next acquire.
  if (Slot.isImplicit()) {
    LLVM_DEBUG(dbgs() << "Released implicit job slot.\n");
    [[maybe_unused]] bool was_already_released =
        HasImplicitSlot.exchange(true, std::memory_order_release);
    assert(!was_already_released && "Implicit slot released twice");
    return;
  }

  uint8_t Token = Slot.getExplicitValue();
  LLVM_DEBUG(dbgs() << "Releasing explicit token '" << (char)Token << "' to FD "
                    << WriteFD << ".\n");

  // For FIFO-based jobservers, the write FD might not be open yet.
  // Open it on the first release.
  if (WriteFD < 0) {
    LLVM_DEBUG(dbgs() << "WriteFD is invalid, opening FIFO: " << FifoPath
                      << "\n");
    WriteFD = ::open(FifoPath.c_str(), O_WRONLY | O_CLOEXEC);
    if (WriteFD < 0) {
      LLVM_DEBUG(dbgs() << "Failed to open FIFO for writing.\n");
      return;
    }
    LLVM_DEBUG(dbgs() << "Opened FIFO as new WriteFD: " << WriteFD << "\n");
  }

  ssize_t Written;
  // Loop to retry on EINTR (interrupted system call).
  do {
    Written = ::write(WriteFD, &Token, 1);
  } while (Written < 0 && errno == EINTR);

  if (Written <= 0) {
    LLVM_DEBUG(dbgs() << "Failed to write token to pipe, write returned "
                      << Written << "\n");
  }
}
