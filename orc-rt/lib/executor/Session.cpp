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

#include <future>

namespace orc_rt {

Session::~Session() { waitForShutdown(); }

void Session::shutdown(OnShutdownCompleteFn OnShutdownComplete) {
  std::vector<std::unique_ptr<ResourceManager>> ToShutdown;

  {
    std::scoped_lock<std::mutex> Lock(M);
    std::swap(ResourceMgrs, ToShutdown);
  }

  shutdownNext(std::move(OnShutdownComplete), Error::success(),
               std::move(ToShutdown));
}

void Session::waitForShutdown() {
  std::promise<void> P;
  auto F = P.get_future();

  shutdown([P = std::move(P)]() mutable { P.set_value(); });

  F.wait();
}

void Session::shutdownNext(
    OnShutdownCompleteFn OnComplete, Error Err,
    std::vector<std::unique_ptr<ResourceManager>> RemainingRMs) {
  if (Err)
    reportError(std::move(Err));

  if (RemainingRMs.empty())
    return OnComplete();

  auto NextRM = std::move(RemainingRMs.back());
  RemainingRMs.pop_back();
  NextRM->shutdown([this, RemainingRMs = std::move(RemainingRMs),
                    OnComplete = std::move(OnComplete)](Error Err) mutable {
    shutdownNext(std::move(OnComplete), std::move(Err),
                 std::move(RemainingRMs));
  });
}

} // namespace orc_rt
