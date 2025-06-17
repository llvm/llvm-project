//===- llvm/Support/Jobserver.cpp - Jobserver Client Implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Jobserver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <new>

#define DEBUG_TYPE "jobserver"

using namespace llvm;

namespace {
struct JobserverConfig {
  enum Mode {
    None,
    PosixFifo,
    PosixPipe,
    Win32Semaphore,
  };
  Mode TheMode = None;
  std::string Path;
  int ReadFD = -1;
  int WriteFD = -1;
};
} // namespace

namespace {
Expected<JobserverConfig> parseNativeMakeFlags(StringRef MakeFlags);
} // namespace

class JobserverClientImpl : public JobserverClient {
  bool IsInitialized = false;
  std::atomic<bool> HasImplicitSlot{true};
  unsigned NumJobs = 0;

public:
  JobserverClientImpl(const JobserverConfig &Config);
  ~JobserverClientImpl() override;

  JobSlot tryAcquire() override;
  void release(JobSlot Slot) override;
  unsigned getNumJobs() const override { return NumJobs; }

  bool isValid() const { return IsInitialized; }

private:
#if defined(LLVM_ON_UNIX)
  int ReadFD = -1;
  int WriteFD = -1;
  std::string FifoPath;
#elif defined(_WIN32)
  void *Semaphore = nullptr;
#endif
};

// Include the platform-specific parts of the class.
#if defined(LLVM_ON_UNIX)
#include "Unix/Jobserver.inc"
#elif defined(_WIN32)
#include "Windows/Jobserver.inc"
#endif

JobserverClient::~JobserverClient() = default;

uint8_t JobSlot::getExplicitValue() const {
  assert(isExplicit() && "Cannot get value of implicit or invalid slot");
  return static_cast<uint8_t>(Value);
}

static std::once_flag GJobserverOnceFlag;
static std::unique_ptr<JobserverClient> GJobserver;

/// This is the main entry point for acquiring a jobserver client. It uses a
/// std::call_once to ensure the singleton `GJobserver` instance is created
/// safely in a multi-threaded environment. On first call, it reads the
/// `MAKEFLAGS` environment variable, parses it, and attempts to construct and
/// initialize a `JobserverClientImpl`. If successful, the global instance is
/// stored in `GJobserver`. Subsequent calls will return the existing instance.
JobserverClient *JobserverClient::getInstance() {
  std::call_once(GJobserverOnceFlag, []() {
    LLVM_DEBUG(
        dbgs()
        << "JobserverClient::getInstance() called for the first time.\n");
    const char *MakeFlagsEnv = getenv("MAKEFLAGS");
    if (!MakeFlagsEnv) {
      errs() << "Warning: failed to create jobserver client due to MAKEFLAGS "
                "environment variable not found\n";
      return;
    }

    LLVM_DEBUG(dbgs() << "Found MAKEFLAGS = \"" << MakeFlagsEnv << "\"\n");

    auto ConfigOrErr = parseNativeMakeFlags(MakeFlagsEnv);
    if (Error Err = ConfigOrErr.takeError()) {
      errs() << "Warning: failed to create jobserver client due to invalid "
                "MAKEFLAGS environment variable: "
             << toString(std::move(Err)) << "\n";
      return;
    }

    JobserverConfig Config = *ConfigOrErr;
    if (Config.TheMode == JobserverConfig::None) {
      errs() << "Warning: failed to create jobserver client due to jobserver "
                "mode missing in MAKEFLAGS environment variable\n";
      return;
    }

    if (Config.TheMode == JobserverConfig::PosixPipe) {
#if defined(LLVM_ON_UNIX)
      if (!areFdsValid(Config.ReadFD, Config.WriteFD)) {
        errs() << "Warning: failed to create jobserver client due to invalid "
                  "Pipe FDs in MAKEFLAGS environment variable\n";
        return;
      }
#endif
    }

    auto Client = std::make_unique<JobserverClientImpl>(Config);
    if (Client->isValid()) {
      LLVM_DEBUG(dbgs() << "Jobserver client created successfully!\n");
      GJobserver = std::move(Client);
    } else
      errs() << "Warning: jobserver client initialization failed.\n";
  });
  return GJobserver.get();
}

/// For testing purposes only. This function resets the singleton instance by
/// destroying the existing client and re-initializing the `std::once_flag`.
/// This allows tests to simulate the first-time initialization of the
/// jobserver client multiple times.
void JobserverClient::resetForTesting() {
  GJobserver.reset();
  // Re-construct the std::once_flag in place to reset the singleton state.
  new (&GJobserverOnceFlag) std::once_flag();
}

namespace {
/// A helper function that checks if `Input` starts with `Prefix`.
/// If it does, it removes the prefix from `Input`, assigns the remainder to
/// `Value`, and returns true. Otherwise, it returns false.
bool getPrefixedValue(StringRef Input, StringRef Prefix, StringRef &Value) {
  if (Input.consume_front(Prefix)) {
    Value = Input;
    return true;
  }
  return false;
}

/// A helper function to parse a string in the format "R,W" where R and W are
/// non-negative integers representing file descriptors. It populates the
/// `ReadFD` and `WriteFD` output parameters. Returns true on success.
bool getFileDescriptorPair(StringRef Input, int &ReadFD, int &WriteFD) {
  if (sscanf(Input.str().c_str(), "%d,%d", &ReadFD, &WriteFD) != 2)
    return false;
  return ReadFD >= 0 && WriteFD >= 0;
}

/// Parses the `MAKEFLAGS` environment variable string to find jobserver
/// arguments. It splits the string into space-separated arguments and searches
/// for `--jobserver-auth` or `--jobserver-fds`. Based on the value of these
/// arguments, it determines the jobserver mode (Pipe, FIFO, or Semaphore) and
/// connection details (file descriptors or path).
Expected<JobserverConfig> parseNativeMakeFlags(StringRef MakeFlags) {
  JobserverConfig Config;
  if (MakeFlags.empty())
    return Config;

  // Split the MAKEFLAGS string into arguments.
  SmallVector<StringRef, 8> Args;
  StringRef S = MakeFlags;
  while (!S.empty()) {
    size_t Start = S.find_first_not_of(" \t");
    if (Start == StringRef::npos)
      break;
    S = S.substr(Start);
    size_t End = S.find_first_of(" \t");
    if (End == StringRef::npos) {
      Args.push_back(S);
      break;
    }
    Args.push_back(S.substr(0, End));
    S = S.substr(End);
  }

  // If '-n' (dry-run) is present as a legacy flag (not starting with '-'),
  // disable the jobserver.
  if (!Args.empty() && !Args[0].starts_with("-") && Args[0].contains('n'))
    return Config;

  // Iterate through arguments to find jobserver flags.
  // Note that make may pass multiple --jobserver-auth flags; the last one wins.
  for (StringRef Arg : Args) {
    StringRef Value;
    if (getPrefixedValue(Arg, "--jobserver-auth=", Value)) {
      int R, W;
      // Try to parse as a file descriptor pair first.
      if (getFileDescriptorPair(Value, R, W)) {
        Config.TheMode = JobserverConfig::PosixPipe;
        Config.ReadFD = R;
        Config.WriteFD = W;
      } else {
        StringRef FifoPath;
        // If not FDs, try to parse as a named pipe (fifo).
        if (getPrefixedValue(Value, "fifo:", FifoPath)) {
          Config.TheMode = JobserverConfig::PosixFifo;
          Config.Path = FifoPath.str();
        } else {
          // Otherwise, assume it's a Windows semaphore.
          Config.TheMode = JobserverConfig::Win32Semaphore;
          Config.Path = Value.str();
        }
      }
    } else if (getPrefixedValue(Arg, "--jobserver-fds=", Value)) {
      // This is an alternative, older syntax for the pipe-based server.
      int R, W;
      if (getFileDescriptorPair(Value, R, W)) {
        Config.TheMode = JobserverConfig::PosixPipe;
        Config.ReadFD = R;
        Config.WriteFD = W;
      } else {
        return createStringError(inconvertibleErrorCode(),
                                 "Invalid file descriptor pair in MAKEFLAGS");
      }
    }
  }

// Perform platform-specific validation.
#ifdef _WIN32
  if (Config.TheMode == JobserverConfig::PosixFifo ||
      Config.TheMode == JobserverConfig::PosixPipe)
    return createStringError(
        inconvertibleErrorCode(),
        "FIFO/Pipe-based jobserver is not supported on Windows");
#else
  if (Config.TheMode == JobserverConfig::Win32Semaphore)
    return createStringError(
        inconvertibleErrorCode(),
        "Semaphore-based jobserver is not supported on this platform");
#endif
  return Config;
}
} // namespace
