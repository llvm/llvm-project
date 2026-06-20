//===------------------- JobManager.cpp - LLVM Advisor -------------------===//
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

#include "Core/JobManager.h"
#include "Utils/Hashing.h"
#include "Utils/JSON.h"

using namespace llvm;
using namespace llvm::advisor;

JobManager::JobManager(StorageManager &Storage, unsigned MaxInFlight)
    : Storage(Storage), Pool(hardware_concurrency()),
      MaxInFlight(MaxInFlight) {}

Expected<JobRecord>
JobManager::submit(StringRef Type, json::Value Request,
                   unique_function<Error(CancellationToken &)> Work) {
  if (Queued.load() >= MaxInFlight)
    return createStringError(inconvertibleErrorCode(), "job queue is full");

  JobRecord Job;
  unsigned Seq = NextJobID.fetch_add(1);
  Job.ID = "job_" +
           hashString((Twine(Type) + stringifyJSON(Request) + Twine(Seq)).str());
  Job.Current = JobRecord::Queued;
  Job.Message = Type.str();
  if (Error Err = Storage.metadata().putJob(Job))
    return std::move(Err);

  ++Queued;
  std::shared_ptr<CancellationToken> Token =
      std::make_shared<CancellationToken>();
  Pool.async([this, Job, Token, Work = std::move(Work)]() mutable {
    {
      std::lock_guard<std::mutex> Guard(TokensLock);
      Tokens[Job.ID] = Token;
    }

    if (Error Err = transition(Job.ID, JobRecord::Running, Job.Message)) {
      consumeError(std::move(Err));
      removeJobTracking(Job.ID);
      return;
    }

    Error Err = Token->isCancelled() ? Error::success() : Work(*Token);
    removeJobTracking(Job.ID);

    if (Token->isCancelled()) {
      consumeError(transition(Job.ID, JobRecord::Cancelled, "cancelled"));
      return;
    }

    if (Err) {
      std::string Message = toString(std::move(Err));
      consumeError(transition(Job.ID, JobRecord::Failed, Message));
      return;
    }
    consumeError(transition(Job.ID, JobRecord::Succeeded, Job.Message));
  });

  return Job;
}

void JobManager::removeJobTracking(StringRef JobID) {
  --Queued;
  std::lock_guard<std::mutex> Guard(TokensLock);
  Tokens.erase(JobID);
}

Error JobManager::transition(StringRef JobID, JobRecord::State State,
                             StringRef Message) {
  Expected<JobRecord> Job = Storage.metadata().getJob(JobID);
  if (!Job)
    return Job.takeError();
  Job->Current = State;
  Job->Message = Message.str();
  return Storage.metadata().putJob(*Job);
}

Error JobManager::cancel(StringRef JobID) {
  {
    std::lock_guard<std::mutex> Guard(TokensLock);
    StringMap<std::shared_ptr<CancellationToken>>::iterator I =
        Tokens.find(JobID);
    if (I != Tokens.end())
      I->second->cancel();
  }
  return transition(JobID, JobRecord::Cancelled, "cancelled");
}

SmallVector<JobRecord, 16> JobManager::list() const {
  return Storage.metadata().listJobs();
}
