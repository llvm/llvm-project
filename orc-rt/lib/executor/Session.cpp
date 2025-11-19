//===- Session.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of the Session class and related APIs.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/Session.h"

namespace orc_rt {

Session::~Session() { waitForShutdown(); }

void Session::shutdown(OnShutdownCompleteFn OnShutdownComplete) {
  std::vector<std::unique_ptr<ResourceManager>> ToShutdown;

  {
    std::scoped_lock<std::mutex> Lock(M);
    ShutdownCallbacks.push_back(std::move(OnShutdownComplete));

    // If somebody else has already called shutdown then there's nothing further
    // for us to do here.
    if (State >= SessionState::ShuttingDown)
      return;

    State = SessionState::ShuttingDown;
    std::swap(ResourceMgrs, ToShutdown);
  }

  shutdownNext(Error::success(), std::move(ToShutdown));
}

void Session::waitForShutdown() {
  shutdown([]() {});
  std::unique_lock<std::mutex> Lock(M);
  StateCV.wait(Lock, [&]() { return State == SessionState::Shutdown; });
}

void Session::shutdownNext(
    Error Err, std::vector<std::unique_ptr<ResourceManager>> RemainingRMs) {
  if (Err)
    reportError(std::move(Err));

  if (RemainingRMs.empty())
    return shutdownComplete();

  auto NextRM = std::move(RemainingRMs.back());
  RemainingRMs.pop_back();
  NextRM->shutdown(
      [this, RemainingRMs = std::move(RemainingRMs)](Error Err) mutable {
        shutdownNext(std::move(Err), std::move(RemainingRMs));
      });
}

void Session::shutdownComplete() {

  std::unique_ptr<TaskDispatcher> TmpDispatcher;
  std::vector<OnShutdownCompleteFn> TmpShutdownCallbacks;
  {
    std::lock_guard<std::mutex> Lock(M);
    TmpDispatcher = std::move(Dispatcher);
    TmpShutdownCallbacks = std::move(ShutdownCallbacks);
  }

  TmpDispatcher->shutdown();

  for (auto &OnShutdownComplete : TmpShutdownCallbacks)
    OnShutdownComplete();

  {
    std::lock_guard<std::mutex> Lock(M);
    State = SessionState::Shutdown;
  }
  StateCV.notify_all();
}

} // namespace orc_rt
