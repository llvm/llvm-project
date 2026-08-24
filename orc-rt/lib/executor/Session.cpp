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
#include "orc-rt-c/Logging.h"
#include "orc-rt-c/Session.h"

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

Session::Session(ExecutorProcessInfo EPI, DispatchFn Dispatch,
                 ErrorReporterFn ReportError)
    : EPI(std::move(EPI)), Dispatch(std::move(Dispatch)),
      ReportError(std::move(ReportError)),
      Notifiers(createService<NotificationService>()) {
  ORC_RT_LOG(Info, Session, "Session %p constructed", this);
}

Session::~Session() {
  ORC_RT_LOG(Info, Session, "Session %p destructor called", this);
  shutdown();
  ORC_RT_LOG(Info, Session,
             "Session %p destructor waiting for shutdown state...", this);
  std::unique_lock<std::mutex> Lock(M);
  CV.wait(Lock, [&]() {
    return CurrentState == State::Shutdown && TargetState == State::None;
  });
  ORC_RT_LOG(Info, Session, "Session %p destructor complete", this);
}

void Session::doAttach(std::shared_ptr<ControllerAccess> CA, BootstrapInfo BI) {
  assert(CA && "doAttach called with null CA object");

  {
    std::scoped_lock<std::mutex> Lock(M);
    // A controller can only be attached from the start state, with no other
    // operation requested: a Session is attached at most once, and attach must
    // not be called after -- or concurrently with -- detach or shutdown. See
    // the Session::attach contract.
    //
    // TODO: Settle on a policy for contract violations in release builds
    // (probably abort) and apply it here. Without the assertions below a
    // violating attach proceeds, clobbering TargetState and potentially
    // regressing CurrentState from Detached back to Attached.
    assert(CurrentState == State::Start && TargetState == State::None &&
           "attach raced detach / shutdown, or Session already attached");
    assert(std::atomic_load(&this->CA) == nullptr &&
           "ControllerAccess object already attached?");
    std::atomic_store(&this->CA, CA);
    TargetState = State::Attached;
  }

  CA->connect(std::move(BI));

  {
    std::scoped_lock<std::mutex> Lock(M);
    assert(TargetState >= State::Attached || CurrentState >= State::Detached);

    // There are three possibilities that we have to deal with here:
    // 1. Connection succeeded and we're done.
    //
    //    We just need to move to the Attached state, reset TargetState, and
    //    we're done.
    //
    // 2. Connect failed.
    //
    //    In this case connect must have called notifyDisconnected, which
    //    should have initiated the detach. We just need to bail out.
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
    // also Detached or higher then notifyDisconnected must already have been
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

  // Fall through to disconnect from case (3) above.
  CA->disconnect();
}

void Session::detach(OnDetachFn OnDetach) {
  ORC_RT_LOG(Info, Session, "Session %p detach called", this);
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
      // A CA is only ever stored with TargetState raised to Attached, and
      // TargetState is not lowered back to None until CurrentState reaches
      // Attached, so reaching the Start state here implies no CA was attached.
      assert(std::atomic_load(&this->CA) == nullptr &&
             "Start state, but a ControllerAccess is attached?");
      // No controller was ever attached, so the disconnect trivially succeeds.
      proceedToDetach(Lock, nullptr, Error::success());
      return;
    }
  }

  TmpCA->disconnect();
}

void Session::shutdown(OnShutdownFn OnShutdown) {
  ORC_RT_LOG(Info, Session, "Session %p shutdown called", this);
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
      // No controller was ever attached, so the disconnect trivially succeeds.
      proceedToDetach(Lock, nullptr, Error::success());
      return;
    case State::Attached:
      TmpCA = std::atomic_load(&this->CA);
      break;
    case State::Detached:
      Lock.unlock();
      waitForKeepalivesThenShutdown();
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

void Session::handleDisconnect(Error Err) {
  ORC_RT_LOG(Info, Session, "Session %p handle-disconnect", this);
  // If we get here we _don't_ need to call disconnect.
  std::unique_lock<std::mutex> Lock(M);
  assert(CurrentState <= State::Attached);
  TargetState = std::max(TargetState, State::Detached);
  proceedToDetach(Lock, std::atomic_exchange(&this->CA, {}), std::move(Err));
}

void Session::proceedToDetach(std::unique_lock<std::mutex> &Lock,
                              std::shared_ptr<ControllerAccess> TmpCA,
                              Error DisconnectErr) {
  std::vector<Service *> ToNotify;
  ToNotify.reserve(Services.size());
  for (auto &Srv : Services)
    ToNotify.push_back(Srv.get());
  bool ShutdownRequested = TargetState == State::Shutdown;
  CurrentState = State::Detached;
  Lock.unlock();

  // Report how the controller connection ended: to the on-disconnect handler if
  // one was installed, otherwise via reportError if it ended abnormally. Every
  // detach path funnels through here, so this runs exactly once per Session --
  // including for a Session that never attached, which reports success.
  //
  // This runs without holding M, since it is client code: holding M across it
  // would deadlock any Session call the handler makes. It also runs before the
  // controller is released below, so the handler can rely on the
  // ControllerAccess outliving it, and before Services are notified.
  //
  // The handler is documented to be able to call shutdown. That works because
  // CurrentState is already Detached and every caller raises TargetState to at
  // least Detached before getting here, so a re-entrant shutdown or detach is
  // absorbed by the early-return in those functions: it registers its callback
  // and raises TargetState, which completeDetach then acts on. Preserve those
  // two properties when reordering this.
  //
  // OnDisconnect is only ever written before attach, so the unlocked read is
  // safe.
  if (OnDisconnect) {
    OnDisconnect(std::move(DisconnectErr));
    // Release anything the handler captured now, rather than holding it until
    // the Session is destroyed. Exactly-once is structural (proceedToDetach
    // runs once), so this is not guarding against a second call.
    OnDisconnect = {};
  } else if (DisconnectErr)
    reportError(std::move(DisconnectErr));

  // Throw away controller if present.
  TmpCA.reset();

  // Notify services.
  ORC_RT_LOG(Debug, Session, "Session %p detaching services", this);
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

  waitForKeepalivesThenShutdown();
}

void Session::waitForKeepalivesThenShutdown() {
  ORC_RT_LOG(Info, Session, "Session %p waiting for keepalives", this);
  KeepaliveTaskGroup->addOnComplete([this]() { proceedToShutdown(); });
  KeepaliveTaskGroup->close();
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

  ORC_RT_LOG(Debug, Session, "Session %p shutting down services", this);
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
  ORC_RT_LOG(Info, Session, "Session %p completing shutdown", this);
  {
    std::scoped_lock<std::mutex> Lock(M);
    assert(CurrentState == State::Shutdown);
    assert(TargetState == State::Shutdown);
    TargetState = State::None;
  }
  CV.notify_all();
}

void Session::sendWrapperResult(WrapperFunctionBuffer ResultBytes,
                                uint64_t CallId) {
  if (auto TmpCA = std::atomic_load(&CA))
    TmpCA->sendWrapperResult(std::move(ResultBytes), CallId);
}

void Session::wrapperReturn(orc_rt_SessionRef S,
                            orc_rt_WrapperFunctionBuffer ResultBytes,
                            uint64_t CallId) {
  unwrap(S)->sendWrapperResult(WrapperFunctionBuffer(ResultBytes), CallId);
}

// --- C API Implementation ---

extern "C" void orc_rt_Session_callController(
    orc_rt_SessionRef S, orc_rt_ControllerHandlerTag T,
    orc_rt_WrapperFunctionBuffer ArgBytes,
    orc_rt_Session_CallControllerReturn Return, void *ReturnCtx) {
  unwrap(S)->callController(
      [S, Return, ReturnCtx](WrapperFunctionBuffer ResultBytes) {
        Return(S, ResultBytes.release(), ReturnCtx);
      },
      T, WrapperFunctionBuffer(ArgBytes));
}

} // namespace orc_rt
