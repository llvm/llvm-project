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

Session::ControllerAccess::~ControllerAccess() = default;

Session::~Session() { waitForShutdown(); }

void Session::shutdown(OnShutdownCompleteFn OnShutdownComplete) {
  // Safe to call concurrently / redundantly.
  detachFromController();

  {
    std::scoped_lock<std::mutex> Lock(M);
    if (SI) {
      SI->OnCompletes.push_back(std::move(OnShutdownComplete));
      return;
    }

    SI = std::make_unique<ShutdownInfo>();
    SI->OnCompletes.push_back(std::move(OnShutdownComplete));
    std::swap(SI->ResourceMgrs, ResourceMgrs);
  }

  shutdownNext(Error::success());
}

void Session::waitForShutdown() {
  shutdown([]() {});
  std::unique_lock<std::mutex> Lock(M);
  SI->CompleteCV.wait(Lock, [&]() { return SI->Complete; });
}

void Session::addResourceManager(std::unique_ptr<ResourceManager> RM) {
  std::scoped_lock<std::mutex> Lock(M);
  assert(!SI && "addResourceManager called after shutdown");
  ResourceMgrs.push_back(std::move(RM));
}

void Session::setController(std::shared_ptr<ControllerAccess> CA) {
  assert(CA && "Cannot attach null controller");
  std::scoped_lock<std::mutex> Lock(M);
  assert(!this->CA && "Cannot re-attach controller");
  assert(!SI && "Cannot attach controller after shutdown");
  this->CA = std::move(CA);
}

void Session::detachFromController() {
  if (auto TmpCA = CA) {
    TmpCA->doDisconnect();
    CA = nullptr;
  }
}

void Session::shutdownNext(Error Err) {
  if (Err)
    reportError(std::move(Err));

  if (SI->ResourceMgrs.empty())
    return shutdownComplete();

  // Get the next ResourceManager to shut down.
  auto NextRM = std::move(SI->ResourceMgrs.back());
  SI->ResourceMgrs.pop_back();
  NextRM->shutdown([this](Error Err) { shutdownNext(std::move(Err)); });
}

void Session::shutdownComplete() {

  std::unique_ptr<TaskDispatcher> TmpDispatcher;
  {
    std::lock_guard<std::mutex> Lock(M);
    TmpDispatcher = std::move(Dispatcher);
  }

  TmpDispatcher->shutdown();

  for (auto &OnShutdownComplete : SI->OnCompletes)
    OnShutdownComplete();

  {
    std::lock_guard<std::mutex> Lock(M);
    SI->Complete = true;
  }

  SI->CompleteCV.notify_all();
}

void Session::wrapperReturn(orc_rt_SessionRef S, uint64_t CallId,
                            orc_rt_WrapperFunctionBuffer ResultBytes) {
  unwrap(S)->sendWrapperResult(CallId, WrapperFunctionBuffer(ResultBytes));
}

} // namespace orc_rt
