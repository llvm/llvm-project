//===-------- Session.h - Session class and related APIs  -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Session class and related APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SESSION_H
#define ORC_RT_SESSION_H

#include "orc-rt/BootstrapInfo.h"
#include "orc-rt/Error.h"
#include "orc-rt/ExecutorProcessInfo.h"
#include "orc-rt/LockedAccess.h"
#include "orc-rt/Service.h"
#include "orc-rt/SimpleSymbolTable.h"
#include "orc-rt/TaskDispatcher.h"
#include "orc-rt/TaskGroup.h"
#include "orc-rt/WrapperFunction.h"
#include "orc-rt/move_only_function.h"

#include "orc-rt-c/CoreTypes.h"
#include "orc-rt-c/WrapperFunction.h"

#include <cassert>
#include <future>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

namespace orc_rt {
class Session;

inline orc_rt_SessionRef wrap(Session *S) noexcept {
  return reinterpret_cast<orc_rt_SessionRef>(S);
}

inline Session *unwrap(orc_rt_SessionRef S) noexcept {
  return reinterpret_cast<Session *>(S);
}

/// Represents an ORC executor Session.
class Session {
private:
  // Implementation helper for callManagedCodeSync (non-void version).
  template <typename RetT> struct ManagedCodeSyncCaller {
    template <typename FnT, typename... ArgTs>
    static std::optional<RetT> call(TaskGroup::Token Tok, FnT &&Fn,
                                    ArgTs &&...Args) {
      if (!Tok)
        return std::nullopt;
      return std::forward<FnT>(Fn)(std::forward<ArgTs>(Args)...);
    }
  };

  // Implementation helper for callManagedCodeSync (void version).
  template <> struct ManagedCodeSyncCaller<void> {
    template <typename FnT, typename... ArgTs>
    static bool call(TaskGroup::Token Tok, FnT &&Fn, ArgTs &&...Args) {
      if (!Tok)
        return false;
      std::forward<FnT>(Fn)(std::forward<ArgTs>(Args)...);
      return true;
    }
  };

  template <typename ReturnArgTupleT> struct ManagedCodeAsyncCaller;

  // Implementation helper for callManagedCodeAsync (non-void version).
  template <typename T>
  struct ManagedCodeAsyncCaller<std::tuple<std::optional<T>>> {
    template <typename ReturnT, typename FnT, typename... ArgTs>
    static void call(TaskGroup::Token Tok, ReturnT &&Return, FnT &&Fn,
                     ArgTs &&...Args) {
      if (!Tok)
        return std::forward<ReturnT>(Return)(std::nullopt);

      std::forward<FnT>(Fn)([Tok = std::move(Tok), R = std::move(Return)](
                                T Value) { R(std::move(Value)); },
                            std::forward<ArgTs>(Args)...);
    }
  };

  // Implementation helper for callManagedCodeAsync (void version).
  template <> struct ManagedCodeAsyncCaller<std::tuple<bool>> {
    template <typename ReturnT, typename FnT, typename... ArgTs>
    static void call(TaskGroup::Token Tok, ReturnT &&Return, FnT &&Fn,
                     ArgTs &&...Args) {
      if (!Tok)
        return std::forward<ReturnT>(Return)(false);

      std::forward<FnT>(Fn)(
          [Tok = std::move(Tok), R = std::move(Return)]() { R(true); },
          std::forward<ArgTs>(Args)...);
    }
  };

public:
  using ErrorReporterFn = move_only_function<void(Error)>;
  using OnDetachFn = move_only_function<void()>;
  using OnShutdownFn = move_only_function<void()>;

  using HandlerTag = void *;
  using OnCallHandlerCompleteFn =
      move_only_function<void(WrapperFunctionBuffer)>;

  /// Provides access to the controller.
  class ControllerAccess {
    friend class Session;

  public:
    virtual ~ControllerAccess();

  protected:
    using HandlerTag = Session::HandlerTag;
    using OnCallHandlerCompleteFn = Session::OnCallHandlerCompleteFn;

    ControllerAccess(Session &S) : S(S) {}

    /// Initiate connection with controller.
    ///
    /// This will be called by the Session once it is ready to accept requests
    /// from the controller.
    ///
    /// ControllerAccess implementations must not call handleWrapperCall prior
    /// to connect being called.
    ///
    /// Note: The Session may call into the controller (via callController)
    /// during connect, but only in response to a controller-initiated wrapper
    /// call. Callers of Session::attach must not race attach with calls to
    /// Session::callController.
    ///
    /// If connect fails to establish communication with the controller,
    /// ControllerAccess implementations must call notifyDisconnected before
    /// returning from connect.
    virtual void connect(BootstrapInfo BI) = 0;

    /// Initiate disconnection from the controller.
    ///
    /// The Session will call this method at most once to request disconnection
    /// from the controller. However, disconnection may also be initiated by
    /// the controller itself (e.g. a network socket dropping out), potentially
    /// concurrently with a Session-initiated disconnect call.
    ///
    /// ControllerAccess implementations are responsible for handling such
    /// double-sided disconnection gracefully, and must ensure that
    /// notifyDisconnected is called exactly once regardless of how
    /// disconnection occurs. In particular, if the ControllerAccess detects
    /// controller-initiated disconnection and calls notifyDisconnected, it
    /// must tolerate a subsequent or concurrent call to disconnect (which
    /// should be treated as a no-op).
    ///
    /// notifyDisconnected may be called from within disconnect or
    /// asynchronously after disconnect returns. This allows disconnect itself
    /// to be a cheap operation (e.g. signaling a shutdown flag) with the
    /// actual disconnection and notifyDisconnected call happening on another
    /// thread.
    virtual void disconnect() = 0;

    /// Report an error to the session.
    void reportError(Error Err) { S.reportError(std::move(Err)); }

    /// Call the handler in the controller associated with the given tag.
    virtual void callController(OnCallHandlerCompleteFn OnComplete,
                                HandlerTag T,
                                WrapperFunctionBuffer ArgBytes) = 0;

    /// Send the result of the given wrapper function call to the controller.
    virtual void sendWrapperResult(uint64_t CallId,
                                   WrapperFunctionBuffer ResultBytes) = 0;

    /// Notify the Session that the controller has disconnected.
    ///
    /// ControllerAccess implementations must call this method exactly once
    /// when the controller disconnects, whether initiated by a call to
    /// disconnect, by the controller, or by a communication failure.
    ///
    /// It is the ControllerAccess implementation's responsibility to ensure
    /// exactly-once semantics for this method, even when disconnect is called
    /// concurrently with controller-initiated disconnection.
    ///
    /// No calls should be made to reportError or handleWrapperCall after this
    /// method is called.
    void notifyDisconnected() { S.handleDisconnect(); }

    /// Ask the Session to run the given wrapper function.
    ///
    /// Subclasses must not call this method after notifyDisconnected is called.
    void handleWrapperCall(uint64_t CallId, orc_rt_WrapperFunction Fn,
                           WrapperFunctionBuffer ArgBytes) {
      S.handleWrapperCall(CallId, Fn, std::move(ArgBytes));
    }

  private:
    Session &S;
  };

  /// Create a session object. The ReportError function will be called to
  /// report errors generated while serving JIT'd code, e.g. if a memory
  /// management request cannot be fulfilled. (Errors within the JIT'd
  /// program are not generally visible to ORC-RT, but can optionally be
  /// reported by calling the orc_rt_Session_reportError function.)
  ///
  /// Note that entry into the reporter is not synchronized: it may be
  /// called from multiple threads concurrently.
  Session(ExecutorProcessInfo EPI, std::unique_ptr<TaskDispatcher> Dispatcher,
          ErrorReporterFn ReportError);

  // Sessions are not copyable or moveable.
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&) = delete;
  Session &operator=(Session &&) = delete;

  ~Session();

  /// Provides information about the host process that the Session is running
  /// in.
  const ExecutorProcessInfo &processInfo() const noexcept { return EPI; }

  /// Dispatch a task using the Session's TaskDispatcher.
  void dispatch(std::unique_ptr<Task> T) { Dispatcher->dispatch(std::move(T)); }

  /// Report an error via the ErrorReporter function.
  void reportError(Error Err) { ReportError(std::move(Err)); }

  /// Add a Service to the session.
  template <typename ServiceT>
  ServiceT &addService(std::unique_ptr<ServiceT> Srv) {
    assert(Srv && "addService called with null value");
    ServiceT &Ref = *Srv;
    appendService(std::move(Srv));
    return Ref;
  }

  /// Construct an instance of ServiceT from the given arguments and add it to
  /// the Session.
  template <typename ServiceT, typename... ArgTs>
  ServiceT &createService(ArgTs &&...Args) {
    return addService(std::make_unique<ServiceT>(std::forward<ArgTs>(Args)...));
  }

  /// Try to create an instance of ServiceT by forwarding the given arguments
  /// to ServiceT::Create method, which must return an
  /// Expected<std::unique_ptr<ServiceT>>.
  ///
  /// On success, adds the service and returns a reference to it.
  /// On failure returns the Error produced by ServiceT::Create.
  template <typename ServiceT, typename... ArgTs>
  Expected<ServiceT &> tryCreateService(ArgTs &&...Args) {
    auto Srv = ServiceT::Create(std::forward<ArgTs>(Args)...);
    if (!Srv)
      return Srv.takeError();
    return addService(std::move(*Srv));
  }

  /// Initiate connection with controller, using the given BootstrapInfo.
  ///
  /// Upon first call, assuming that the Session has not already been detached
  /// or shutdown, this will take (shared) ownership of CA and call its connect
  /// method.
  ///
  /// If detach or shutdown have already been called then this method will not
  /// take ownership of CA or call its connect method.
  void attach(std::shared_ptr<ControllerAccess> CA, BootstrapInfo BI);

  /// Initiate detach from the controller.
  ///
  /// Signals that controller access is permanently unavailable and notifies
  /// all Services via onDetach. If a controller is attached, this will
  /// request disconnection first.
  ///
  /// The optional OnDetach callback will be called once the detach is
  /// complete.
  ///
  /// If the Session is already detached or shut down, the callback (if
  /// provided) will be called immediately.
  void detach(OnDetachFn OnDetach = {});

  /// Initiate session shutdown.
  ///
  /// Shutdown proceeds through the following phases:
  ///   1. Detach: If not already detached, disconnects the controller and
  ///      notifies all Services via onDetach.
  ///   2. Drain: Waits for all in-flight managed code calls to complete
  ///      (via ManagedCodeCallsGroup).
  ///   3. Shutdown services: Calls onShutdown on all Services in reverse
  ///      order.
  ///   4. Shutdown TaskDispatcher.
  ///
  /// The optional OnShutdown callback is called after step (3), before
  /// the TaskDispatcher is shut down.
  void shutdown(OnShutdownFn OnShutdown = {});

  /// Initiate session shutdown and block until complete.
  void waitForShutdown();

  /// Register a callback to be called when the Session detaches from the
  /// controller. If the Session has already detached, the callback will be
  /// called immediately.
  void addOnDetach(OnDetachFn OnDetach);

  /// Register a callback to be called when the Session shuts down. If the
  /// Session has already shut down, the callback will be called immediately.
  void addOnShutdown(OnShutdownFn OnShutdown);

  /// Returns a reference to this Session's ManagedCodeCallsGroup.
  ///
  /// When calling code managed by a Session (e.g. JIT'd code, or library code
  /// loaded on behalf of JIT'd code), clients should hold a token for this
  /// group. That token will prevent the Session from shutting down any Services
  /// (and the Session itself) until calls into managed code have completed.
  ///
  /// Clients should prefer using the callManagedCodeSync and
  /// callManagedCodeAsync helpers to automatically acquire and hold a token
  /// for the duration of a call.
  const std::shared_ptr<TaskGroup> &managedCodeCallsGroup() const {
    return ManagedCodeCallsGroup;
  }

  /// Synchronously call managed code.
  ///
  /// This helper tries to acquire a ManagedCodeCallsGroup token and then call
  /// the given function object with the given arguments while holding the
  /// token.
  ///
  /// If the token is successfully acquired then this function will return the
  /// call result as a std::optional<T> (for a non-void return type T), or
  /// boolean true (for void returns).
  ///
  /// If the token is not successfully acquired then this function will return
  /// std::nullopt (for non-void return type) or boolean false (for void
  /// returns).
  template <typename FnT, typename... ArgTs>
  decltype(auto) callManagedCodeSync(FnT &&Fn, ArgTs &&...Args) {
    return ManagedCodeSyncCaller<std::invoke_result_t<FnT, ArgTs...>>::call(
        TaskGroup::Token(ManagedCodeCallsGroup), std::forward<FnT>(Fn),
        std::forward<ArgTs>(Args)...);
  }

  /// Asynchronously call managed code.
  ///
  /// ReturnT must be a function object that takes either a boolean or a
  /// std::optional<T>.
  ///
  /// callManagedCodeAsync tries to acquire a ManagedCodeCallsGroup token and
  /// then call the given async function object while holding that token.
  ///
  /// If the token is successfully acquired then this function will call Fn,
  /// passing in a wrapped version of Return that takes a T (if Return takes a
  /// std::optional<T>), or a wrapped version of Return that takes no arguments
  /// (if Return takes a bool).
  ///
  /// If the token is not successfully acquired then this function will not
  /// call Fn, but instead immediately call Return with std::nullopt (if Return
  /// takes a std::optional<T>), or false (if Return takes a boolean).
  template <typename ReturnT, typename FnT, typename... ArgTs>
  void callManagedCodeAsync(ReturnT &&Return, FnT &&Fn, ArgTs &&...Args) {
    ManagedCodeAsyncCaller<typename CallableArgInfo<ReturnT>::args_tuple_type>::
        call(TaskGroup::Token(ManagedCodeCallsGroup),
             std::forward<ReturnT>(Return), std::forward<FnT>(Fn),
             std::forward<ArgTs>(Args)...);
  }

  /// Call a tagged handler in the Controller.
  ///
  /// This method can be called directly, but is expected to be more commonly
  /// called via WrapperFunction::call using a CallViaSession object (returned
  /// by the callViaSession method).
  void callController(OnCallHandlerCompleteFn OnComplete, HandlerTag T,
                      WrapperFunctionBuffer ArgBytes) {
    if (auto TmpCA = std::atomic_load(&CA))
      TmpCA->callController(std::move(OnComplete), T, std::move(ArgBytes));
    else
      OnComplete(WrapperFunctionBuffer::createOutOfBandError(
          "no controller attached"));
  }

  /// Provides an async method interface to call, via the given Session, the
  /// controller handler with the given tag.
  ///
  /// Useable as a Caller implementation with WrapperFunction::call.
  class CallViaSession {
  public:
    CallViaSession(Session &S, HandlerTag T) : S(S), T(T) {}

    void operator()(OnCallHandlerCompleteFn &&HandleResult,
                    WrapperFunctionBuffer ArgBytes) {
      S.callController(std::move(HandleResult), T, std::move(ArgBytes));
    }

  private:
    Session &S;
    HandlerTag T;
  };

  /// Get a WrapperFunction::call-compatible Caller that will call through to
  /// the handler with the given tag.
  CallViaSession callViaSession(HandlerTag T) noexcept {
    return CallViaSession(*this, T);
  }

private:
  enum class State {
    /// Used as a placeholder when there is no target state.
    None,

    /// The Session starts in this state.
    Start,

    /// Controller attached.
    Attached,

    /// Controller detached.
    Detached,

    /// Shutdown.
    Shutdown
  };

  class NotificationService;
  NotificationService &addNotificationService();

  void appendService(std::unique_ptr<Service> Srv);

  void handleDisconnect();
  void proceedToDetach(std::unique_lock<std::mutex> &Lock,
                       std::shared_ptr<ControllerAccess> TmpCA);
  void detachServices(std::vector<Service *> ToNotify, bool ShutdownRequested);
  void completeDetach();

  void waitForManagedCodeCallsThenShutdown();
  void proceedToShutdown();
  void shutdownServices(std::vector<Service *> ToNotify);
  void completeShutdown();

  void handleWrapperCall(uint64_t CallId, orc_rt_WrapperFunction Fn,
                         WrapperFunctionBuffer ArgBytes) {
    if (!ManagedCodeCallsGroup->acquireToken()) {
      // The ManagedCodeCallsGroup is only closed after detach, so if token
      // acquisition fails we don't try to return an error: the controller
      // should already have signalled error to the caller, and we have no
      // way to transmit an error anyway.
      return;
    }

    dispatch(makeGenericTask([=, ArgBytes = std::move(ArgBytes)]() mutable {
      Fn(wrap(this), CallId, wrapperReturn, ArgBytes.release());
    }));
  }

  void sendWrapperResult(uint64_t CallId, WrapperFunctionBuffer ResultBytes) {
    if (auto TmpCA = std::atomic_load(&CA))
      TmpCA->sendWrapperResult(CallId, std::move(ResultBytes));
    ManagedCodeCallsGroup->releaseToken();
  }

  static void wrapperReturn(orc_rt_SessionRef S, uint64_t CallId,
                            orc_rt_WrapperFunctionBuffer ResultBytes);

  ExecutorProcessInfo EPI;
  std::unique_ptr<TaskDispatcher> Dispatcher;
  std::shared_ptr<TaskGroup> ManagedCodeCallsGroup = TaskGroup::Create();
  std::shared_ptr<ControllerAccess> CA;
  ErrorReporterFn ReportError;

  mutable std::mutex M;
  State CurrentState = State::Start;
  State TargetState = State::None;
  std::vector<std::unique_ptr<Service>> Services;
  NotificationService &Notifiers;
};

} // namespace orc_rt

#endif // ORC_RT_SESSION_H
