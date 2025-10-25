//===- llvm/Support/Jobserver.h - Jobserver Client --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a client for the GNU Make jobserver protocol. This allows
// LLVM tools to coordinate parallel execution with a parent `make` process.
//
// The jobserver protocol is a mechanism for GNU Make to share its pool of
// available "job slots" with the subprocesses it invokes. This is particularly
// useful for tools that can perform parallel operations themselves (e.g., a
// multi-threaded linker or compiler). By participating in this protocol, a
// tool can ensure the total number of concurrent jobs does not exceed the
// limit specified by the user (e.g., `make -j8`).
//
// How it works:
//
// 1. Establishment:
//    A child process discovers the jobserver by inspecting the `MAKEFLAGS`
//    environment variable. If a jobserver is active, this variable will
//    contain a `--jobserver-auth=<value>` argument. The format of `<value>`
//    determines how to communicate with the server.
//
// 2. The Implicit Slot:
//    Every command invoked by `make` is granted one "implicit" job slot. This
//    means a tool can always perform at least one unit of work without needing
//    to communicate with the jobserver. This implicit slot should NEVER be
//    released back to the jobserver.
//
// 3. Acquiring and Releasing Slots:
//    On POSIX systems, the jobserver is implemented as a pipe. The
//    `--jobserver-auth` value specifies either a path to a named pipe
//    (`fifo:PATH`) or a pair of file descriptors (`R,W`). The pipe is
//    pre-loaded with single-character tokens, one for each available job slot.
//
//    - To acquire an additional slot, a client reads a single-character token
//      from the pipe.
//    - To release a slot, the client must write the *exact same* character
//      token back to the pipe.
//
//    It is critical that a client releases all acquired slots before it exits,
//    even in cases of error, to avoid deadlocking the build.
//
// Example:
//    A multi-threaded linker invoked by `make -j8` wants to use multiple
//    threads. It first checks for the jobserver. It knows it has one implicit
//    slot, so it can use one thread. It then tries to acquire 7 more slots by
//    reading 7 tokens from the jobserver pipe. If it only receives 3 tokens,
//    it knows it can use a total of 1 (implicit) + 3 (acquired) = 4 threads.
//    Before exiting, it must write the 3 tokens it read back to the pipe.
//
// For more context, see:
//   - GNU Make manual on job slots:
//     https://www.gnu.org/software/make/manual/html_node/Job-Slots.html
//   - LLVM RFC discussion on jobserver support:
//     https://discourse.llvm.org/t/rfc-adding-gnu-make-jobserver-
//     support-to-llvm-for-coordinated-parallelism/87034
//   - Ninja’s jobserver support PR:
//     https://github.com/ninja-build/ninja/pull/2506
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_JOBSERVER_H
#define LLVM_SUPPORT_JOBSERVER_H

#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace llvm {

/// A JobSlot represents a single job slot that can be acquired from or released
/// to a jobserver pool. This class is move-only.
class JobSlot {
public:
  /// Default constructor creates an invalid instance.
  JobSlot() = default;

  // Move operations are allowed.
  JobSlot(JobSlot &&Other) noexcept : Value(Other.Value) {
    Other.Value = kInvalidValue;
  }
  JobSlot &operator=(JobSlot &&Other) noexcept {
    if (this != &Other) {
      this->Value = Other.Value;
      Other.Value = kInvalidValue;
    }
    return *this;
  }

  // Copy operations are disallowed.
  JobSlot(const JobSlot &) = delete;
  JobSlot &operator=(const JobSlot &) = delete;

  /// Returns true if this instance is valid (either implicit or explicit).
  bool isValid() const { return Value >= 0; }

  /// Returns true if this instance represents the implicit job slot.
  bool isImplicit() const { return Value == kImplicitValue; }

  static JobSlot createExplicit(uint8_t V) {
    return JobSlot(static_cast<int16_t>(V));
  }

  static JobSlot createImplicit() { return JobSlot(kImplicitValue); }

  uint8_t getExplicitValue() const;
  bool isExplicit() const { return isValid() && !isImplicit(); }

private:
  friend class JobserverClient;
  friend class JobserverClientImpl;

  JobSlot(int16_t V) : Value(V) {}

  /// The jobserver pipe carries explicit tokens (bytes 0–255). We reserve two
  /// sentinels in Value for special cases:
  ///   kInvalidValue  (-1): no slot held
  ///   kImplicitValue (INT16_MAX): implicit slot granted at startup (no pipe
  ///   I/O)
  ///
  /// We use int16_t so Value can store 0–255 explicit tokens and
  /// sentinels without overflow, enforces fixed 16-bit width, and avoids
  /// unsigned/signed mix-ups.
  static constexpr int16_t kInvalidValue = -1;
  static constexpr int16_t kImplicitValue = INT16_MAX;
  int16_t Value = kInvalidValue;
};

/// The public interface for a jobserver client.
/// This client is a lazy-initialized singleton that is created on first use.
class JobserverClient {
public:
  virtual ~JobserverClient();

  /// Tries to acquire a job slot from the pool. On failure (e.g., if the pool
  /// is empty), this returns an invalid JobSlot instance. The first successful
  /// call will always return the implicit slot.
  virtual JobSlot tryAcquire() = 0;

  /// Releases a job slot back to the pool.
  virtual void release(JobSlot Slot) = 0;

  /// Returns the number of job slots available, as determined on first use.
  /// This value is cached. Returns 0 if no jobserver is active.
  virtual unsigned getNumJobs() const = 0;

  /// Returns the singleton instance of the JobserverClient.
  /// The instance is created on the first call to this function.
  /// Returns a nullptr if no jobserver is configured or an error occurs.
  static JobserverClient *getInstance();

  /// Resets the singleton instance. For testing purposes only.
  static void resetForTesting();
};

} // end namespace llvm

#endif // LLVM_SUPPORT_JOBSERVER_H
