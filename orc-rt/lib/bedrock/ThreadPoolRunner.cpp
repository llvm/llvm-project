//===- ThreadPoolRunner.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/bedrock/ThreadPoolRunner.h
// header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/ThreadPoolRunner.h"

#include <cassert>

namespace orc_rt {

ThreadPoolRunner::ThreadPoolRunner(size_t NumThreads) {
  Workers.reserve(NumThreads);
  for (size_t I = 0; I < NumThreads; ++I)
    Workers.emplace_back([this]() { workerLoop(); });
}

ThreadPoolRunner::~ThreadPoolRunner() {
  {
    std::scoped_lock<std::mutex> Lock(M);
    Stop = true;
  }
  CV.notify_all();
  for (auto &Worker : Workers)
    Worker.join();
}

void ThreadPoolRunner::operator()(move_only_function<void()> Task) {
  {
    std::scoped_lock<std::mutex> Lock(M);
    assert(!Stop &&
           "operator() called on ThreadPoolRunner after destruction begun");
    Pending.push_back(std::move(Task));
  }
  CV.notify_one();
}

void ThreadPoolRunner::workerLoop() {
  while (true) {
    move_only_function<void()> Call;
    {
      std::unique_lock<std::mutex> Lock(M);
      CV.wait(Lock, [this]() { return !Pending.empty() || Stop; });

      if (Pending.empty() && Stop)
        return;

      Call = std::move(Pending.back());
      Pending.pop_back();
    }

    Call();
  }
}

} // namespace orc_rt
