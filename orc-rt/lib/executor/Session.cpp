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

class Session::NotificationService : public Service {
public:
  void addOnDetach(Session::OnDetachFn OnDetach) {
    ToNotifyOnDetach.push_back(std::move(OnDetach));
  }

  void addOnShutdown(Session::OnShutdownFn OnShutdown) {
    ToNotifyOnShutdown.push_back(std::move(OnShutdown));
  }

  void onDetach(OnCompleteFn OnComplete, bool ShutdownRequested) override {
    while (!ToNotifyOnDetach.empty()) {
      auto ToNotify = std::move(ToNotifyOnDetach.back());
      ToNotifyOnDetach.pop_back();
      ToNotify();
    }
    OnComplete();
  }

  void onShutdown(OnCompleteFn OnComplete) override {
    while (!ToNotifyOnShutdown.empty()) {
      auto ToNotify = std::move(ToNotifyOnShutdown.back());
      ToNotifyOnShutdown.pop_back();
      ToNotify();
    }
    OnComplete();
  }

private:
  std::vector<Session::OnDetachFn> ToNotifyOnDetach;
  std::vector<Session::OnShutdownFn> ToNotifyOnShutdown;
};

Session::ControllerAccess::~ControllerAccess() = default;

Session::Session(ExecutorProcessInfo EPI,
                 std::unique_ptr<TaskDispatcher> Dispatcher,
                 ErrorReporterFn ReportError)
    : EPI(std::move(EPI)), Dispatcher(std::move(Dispatcher)),
      ReportError(std::move(ReportError)),
      Notifiers(createService<NotificationService>()) {}

Session::~Session() {
  shutdown();
  std::unique_lock<std::mutex> Lock(M);
  CV.wait(Lock, [&]() {
    return CurrentState == State::Shutdown && TargetState == State::None;
  });
}

void Session::attach(std::shared_ptr<ControllerAccess> CA, BootstrapInfo BI) {
  assert(CA && "attach called with null CA object");

  {
    std::scoped_lock<std::mutex> Lock(M);
    // Controller can only be attached from the start state if no
    // other operation has been requested.
    if (CurrentState != State::Start || TargetState != State::None)
      return;
    assert(std::atomic_load(&this->CA) == nullptr &&
           "ControllerAccess object already attached?");
    std::atomic_store(&this->CA, CA);
    TargetState = State::Attached;
  }

  CA->connect(std::move(BI));

  {
    std::scoped_lock<std::mutex> Lock(M);
    assert(TargetState >= State::Attached);

    // There are three possibilities that we have to deal with here:
    // 1. Connection succeeded and we're done.
    //
    //    We just need to move to the Attached state, reset TargetState, and
    //    we're done.
    //
    // 2. Connect failed.
    //
    //    In this case connect must have called handleDisconnect, which should
    //    have initiated the detach. We just need to bail out.
    //
    // 3. Connection succeeded but a detach or shutdown was requested
    //    concurrently. In this case we need to start the detach process.
    //
    // To distinguish between these we first look at the target state. If it's
    // Attached then it's option (1) and we're done:
    if (TargetState == State::Attached) {
      CurrentState = State::Attached;
      TargetState = State::None;
      return;
    }

    // The target state is Detached or higher. Check the current state. If it's
    // also Detached or higher then handleDisconnect must already have been
    // called (in turn calling proceedToDetach, which updated the current
    // state). In this case we're in option (2) and we just need to bail out.
    if (CurrentState >= State::Detached)
      return;

    // The target state is Detached or higher, but the current state is still
    // Start. Someone must have called detach / shutdown concurrently. This is
    // option (3) and we just need to update the current state and run
    // disconnect.
    CurrentState = State::Attached;
  }

  CA->disconnect();
}

void Session::detach(OnDetachFn OnDetach) {
  addOnDetach(std::move(OnDetach));

  std::shared_ptr<ControllerAccess> TmpCA;
  {
    std::unique_lock<std::mutex> Lock(M);

    // Check if someone's already managing transitions.
    if (TargetState != State::None) {
      TargetState = std::max(TargetState, State::Detached);
      return;
    }

    // Nobody's managing transitions, but this request is redundant.
    if (CurrentState >= State::Detached)
      return;

    // We've actually got work to do.
    TargetState = State::Detached;
    assert((CurrentState == State::Start || CurrentState == State::Attached) &&
           "Unexpected current state");

    if (CurrentState == State::Attached) {
      assert(CA && "Attached, but not CA?");
      TmpCA = std::atomic_load(&this->CA);
    } else {
      assert(CurrentState == State::Start);
      proceedToDetach(Lock, std::atomic_exchange(&this->CA, {}));
      return;
    }
  }

  TmpCA->disconnect();
}

void Session::shutdown(OnShutdownFn OnShutdown) {
  addOnShutdown(std::move(OnShutdown));

  std::shared_ptr<ControllerAccess> TmpCA;
  {
    std::unique_lock<std::mutex> Lock(M);

    // Check if someone's already managing transitions.
    if (TargetState != State::None) {
      TargetState = std::max(TargetState, State::Shutdown);
      return;
    }

    // Nobody's managing transition, but this request is redundant.
    if (CurrentState == State::Shutdown)
      return;

    TargetState = State::Shutdown;
    assert((CurrentState == State::Start || CurrentState == State::Attached ||
            CurrentState == State::Detached) &&
           "Unexpected current state");

    switch (CurrentState) {
    case State::Start:
      proceedToDetach(Lock, nullptr);
      return;
    case State::Attached:
      TmpCA = std::atomic_load(&this->CA);
      break;
    case State::Detached:
      Lock.unlock();
      waitForManagedCodeTasksThenShutdown();
      return;
    default:
      assert(false && "Illegal state");
      abort();
    }
  }

  TmpCA->disconnect();
}

void Session::addOnDetach(OnDetachFn OnDetach) {
  if (!OnDetach)
    return;
  {
    std::scoped_lock<std::mutex> Lock(M);
    if (CurrentState < State::Detached) {
      Notifiers.addOnDetach(std::move(OnDetach));
      return;
    }
  }
  // We've already detached. Run in-place.
  OnDetach();
}

void Session::addOnShutdown(OnShutdownFn OnShutdown) {
  if (!OnShutdown)
    return;
  {
    std::scoped_lock<std::mutex> Lock(M);
    if (CurrentState < State::Shutdown) {
      Notifiers.addOnShutdown(std::move(OnShutdown));
      return;
    }
  }
  // We've already shutdown. Run in-place.
  OnShutdown();
}

void Session::appendService(std::unique_ptr<Service> Srv) {

  bool ShuttingDown = false;
  {
    std::scoped_lock<std::mutex> Lock(M);
    if (CurrentState < State::Detached) {
      Services.push_back(std::move(Srv));
      return;
    }
    ShuttingDown = TargetState == State::Shutdown;
  }

  // Already detached. Call onDetach on the service.
  assert(Srv && "Should be non-null here");
  Srv->onDetach([]() {}, ShuttingDown);

  // Try to append again.
  {
    std::scoped_lock<std::mutex> Lock(M);
    if (CurrentState < State::Shutdown) {
      Services.push_back(std::move(Srv));
      return;
    }
  }

  // Already shutdown. Call onShutdown on the service.
  assert(Srv && "Should be non-null here");
  Srv->onShutdown([]() {});

  // At this point the service has already been shut down, but we need to keep
  // the object alive until the Session is destroyed, so append it anyway.
  {
    std::scoped_lock<std::mutex> Lock(M);
    Services.push_back(std::move(Srv));
  }
}

void Session::handleDisconnect() {
  // If we get here we _don't_ need to call disconnect.
  std::unique_lock<std::mutex> Lock(M);
  assert(CurrentState <= State::Attached);
  TargetState = std::max(TargetState, State::Detached);
  proceedToDetach(Lock, std::atomic_exchange(&this->CA, {}));
}

void Session::proceedToDetach(std::unique_lock<std::mutex> &Lock,
                              std::shared_ptr<ControllerAccess> TmpCA) {
  std::vector<Service *> ToNotify;
  ToNotify.reserve(Services.size());
  for (auto &Srv : Services)
    ToNotify.push_back(Srv.get());
  bool ShutdownRequested = TargetState == State::Shutdown;
  CurrentState = State::Detached;
  Lock.unlock();

  // Throw away controller if present.
  TmpCA.reset();

  // Notify services.
  detachServices(std::move(ToNotify), ShutdownRequested);
}

void Session::detachServices(std::vector<Service *> ToNotify,
                             bool ShutdownRequested) {
  if (ToNotify.empty())
    return completeDetach();

  auto *Srv = ToNotify.back();
  ToNotify.pop_back();
  Srv->onDetach(
      [this, ToNotify = std::move(ToNotify), ShutdownRequested]() {
        detachServices(std::move(ToNotify), ShutdownRequested);
      },
      ShutdownRequested);
}

void Session::completeDetach() {
  {
    std::scoped_lock<std::mutex> Lock(M);
    assert(CurrentState == State::Detached);
    if (TargetState == State::Detached) {
      TargetState = State::None;
      return;
    }
    // Someone must have requested shutdown.
    assert(TargetState == State::Shutdown);
  }

  waitForManagedCodeTasksThenShutdown();
}

void Session::waitForManagedCodeTasksThenShutdown() {
  ManagedCodeTaskGroup->addOnComplete([this]() { proceedToShutdown(); });
  ManagedCodeTaskGroup->close();
}

void Session::proceedToShutdown() {
  std::vector<Service *> ToNotify;
  {
    std::scoped_lock<std::mutex> Lock(M);
    ToNotify.reserve(Services.size());
    for (auto &Srv : Services)
      ToNotify.push_back(Srv.get());
    CurrentState = State::Shutdown;
  }

  shutdownServices(std::move(ToNotify));
}

void Session::shutdownServices(std::vector<Service *> ToNotify) {
  if (ToNotify.empty())
    return completeShutdown();

  auto *Srv = ToNotify.back();
  ToNotify.pop_back();
  Srv->onShutdown([this, ToNotify = std::move(ToNotify)]() {
    shutdownServices(std::move(ToNotify));
  });
}

void Session::completeShutdown() {
  Dispatcher->shutdown();

  {
    std::scoped_lock<std::mutex> Lock(M);
    assert(CurrentState == State::Shutdown);
    assert(TargetState == State::Shutdown);
    TargetState = State::None;
  }
  CV.notify_all();
}

void Session::wrapperReturn(orc_rt_SessionRef S, uint64_t CallId,
                            orc_rt_WrapperFunctionBuffer ResultBytes) {
  unwrap(S)->sendWrapperResult(CallId, WrapperFunctionBuffer(ResultBytes));
}

} // namespace orc_rt
