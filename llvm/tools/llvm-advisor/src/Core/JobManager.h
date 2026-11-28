//===------------------- JobManager.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages job queue and executes capability nodes in parallel.
// Provides worker pool and streaming result dispatch.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Core/AdvisorTypes.h"
#include "Storage/StorageManager.h"
#include "llvm/Support/ThreadPool.h"
#include <atomic>
#include <memory>
#include <mutex>

namespace llvm::advisor {

class CancellationToken {
public:
  void cancel() { Cancelled.store(true); }
  bool isCancelled() const { return Cancelled.load(); }

private:
  std::atomic<bool> Cancelled{false};
};

class JobManager {
public:
  /// JobManager must outlive all submitted jobs. Destroying JobManager while
  /// jobs are active is undefined behavior.
  explicit JobManager(StorageManager &Storage, unsigned MaxInFlight = 1000);

  Expected<JobRecord> submit(StringRef Type, json::Value Request,
                             unique_function<Error(CancellationToken &)> Work);
  Error cancel(StringRef JobID);
  SmallVector<JobRecord, 16> list() const;

private:
  Error transition(StringRef JobID, JobRecord::State State, StringRef Message);
  void removeJobTracking(StringRef JobID);

  StorageManager &Storage;
  DefaultThreadPool Pool;
  unsigned MaxInFlight;
  std::atomic<unsigned> Queued{0};
  std::atomic<unsigned> NextJobID{0};
  mutable std::mutex TokensLock;
  StringMap<std::shared_ptr<CancellationToken>> Tokens;
};

} // namespace llvm::advisor
