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
#include "orc-rt/TaskGroup.h"
#include "orc-rt/WrapperFunction.h"
#include "orc-rt/move_only_function.h"

#include "orc-rt-c/CoreTypes.h"
#include "orc-rt-c/WrapperFunction.h"

#include <cassert>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
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
  // Implementation helper for callManagedCode (non-void version).
  template <typename RetT> struct ManagedCodeCaller {
    template <typename FnT, typename... ArgTs>
    static std::optional<RetT> call(TaskGroup::Token Tok, FnT &&Fn,
                                    ArgTs &&...Args) {
      if (!Tok)
        return std::nullopt;
      return std::forward<FnT>(Fn)(std::forward<ArgTs>(Args)...);
    }
  };

  // Implementation helper for callManagedCode (void version).
  template <> struct ManagedCodeCaller<void> {
    template <typename FnT, typename... ArgTs>
    static bool call(TaskGroup::Token Tok, FnT &&Fn, ArgTs &&...Args) {
      if (!Tok)
        return false;
      std::forward<FnT>(Fn)(std::forward<ArgTs>(Args)...);
      return true;
    }
  };

public:
  using ErrorReporterFn = move_only_function<void(Error)>;
  using OnDetachFn = move_only_function<void()>;
  using OnShutdownFn = move_only_function<void()>;

  /// Return value callback used to return results from callController.
  using OnControllerCallReturnFn =
      move_only_function<void(WrapperFunctionBuffer)>;

  /// Callback used by the Session to run incoming wrapper-function calls.
  ///
  /// A ManagedCodeTaskGroup token is created for each call to this callback,
  /// and implementations must eventually call either Fn (typically as
  /// Fn(S, CallId, Return, ArgBytes.release())), or call Return directly to
  /// bail out of the call (typically with
  /// WrapperFunctionBuffer::createOutOfBandError(...)). Failing to do either
  /// will block Session shutdown indefinitely.
  using RunWrapperCall = move_only_function<void(
      orc_rt_SessionRef S, uint64_t CallId, orc_rt_WrapperFunctionReturn Return,
      orc_rt_WrapperFunction Fn, WrapperFunctionBuffer ArgBytes)>;

  /// Tag used to identify executor-callable functions in the controller.
  /// See callController.
  using HandlerTag = void *;

  /// Provides access to the controller.
  class ControllerAccess {
    friend class Session;

  public:
    virtual ~ControllerAccess();

  protected:
    using HandlerTag = Session::HandlerTag;
    using OnControllerCallReturnFn = Session::OnControllerCallReturnFn;

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
    virtual void callController(OnControllerCallReturnFn OnComplete,
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
  /// The RunCall callback will be invoked for every incoming wrapper-function
  /// call, and is responsible for arranging the call to be run (inline,
  /// queued, or posted to a thread pool, at the caller's discretion).
  ///
  /// Note that entry into the reporter is not synchronized: it may be
  /// called from multiple threads concurrently.
  Session(ExecutorProcessInfo EPI, RunWrapperCall RunCall,
          ErrorReporterFn ReportError);

  // Sessions are not copyable or moveable.
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&) = delete;
  Session &operator=(Session &&) = delete;

  /// Destroy the session object.
  ///
  /// This will trigger shutdown if it has not happened already. Destruction
  /// will block until the Session lifecycle completes.
  ~Session();

  /// Provides information about the host process that the Session is running
  /// in.
  const ExecutorProcessInfo &processInfo() const noexcept { return EPI; }

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

  /// Construct a ControllerAccessT and immediately attach using the given
  /// BootstrapInfo.
  ///
  /// This enables one-line attach operations in the common case where the
  /// ControllerAccess implementation requires no further configuration after
  /// construction and cannot fail to construct. ControllerAccess
  /// implementations whose setup can fail (e.g. binding a socket) should
  /// provide a Create factory and use tryAttach instead.
  ///
  /// ControllerAccessT is constructed with a reference to this Session as its
  /// first argument, followed by the given args, as required by the
  /// ControllerAccess base constructor.
  template <typename ControllerAccessT, typename... ArgTs>
  void attach(BootstrapInfo BI, ArgTs &&...Args) {
    doAttach(std::make_shared<ControllerAccessT>(*this,
                                                 std::forward<ArgTs>(Args)...),
             std::move(BI));
  }

  /// Try to construct a ControllerAccessT by forwarding a reference to this
  /// Session and the given args to ControllerAccessT::Create, which must
  /// return an Expected<std::shared_ptr<ControllerAccessT>>. On success,
  /// immediately attaches using the given BootstrapInfo.
  ///
  /// This is the fallible counterpart to attach<ControllerAccessT>: it allows
  /// ControllerAccess implementations to surface setup failures (e.g. failure
  /// to bind a socket) synchronously as an Error, without ever handing the
  /// caller a usable-but-unconnected ControllerAccess object. Runtime and
  /// remote failures should still be reported asynchronously via
  /// notifyDisconnected.
  ///
  /// ControllerAccessT::Create is passed a reference to this Session as its
  /// first argument, followed by the given args.
  template <typename ControllerAccessT, typename... ArgTs>
  Error tryAttach(BootstrapInfo BI, ArgTs &&...Args) {
    auto CA = ControllerAccessT::Create(*this, std::forward<ArgTs>(Args)...);
    if (!CA)
      return CA.takeError();
    doAttach(std::move(*CA), std::move(BI));
    return Error::success();
  }

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
  ///   2. Drain: Waits for all in-flight tasks accessing managed code to
  ///      complete (via ManagedCodeTaskGroup).
  ///   3. Shutdown services: Calls onShutdown on all Services in reverse
  ///      order.
  ///
  /// The optional OnShutdown callback is called after step (3).
  void shutdown(OnShutdownFn OnShutdown = {});

  /// Register a callback to be called when the Session detaches from the
  /// controller. If the Session has already detached, the callback will be
  /// called immediately.
  void addOnDetach(OnDetachFn OnDetach);

  /// Register a callback to be called when the Session shuts down. If the
  /// Session has already shut down, the callback will be called immediately.
  void addOnShutdown(OnShutdownFn OnShutdown);

  /// Return a TokenSource for this Session's ManagedCodeTaskGroup.
  ///
  /// When calling code managed by a Session (e.g. JIT'd code, or library code
  /// loaded on behalf of JIT'd code), clients should hold a token for this
  /// group, which can be constructed from the returned TokenSource. That token
  /// will delay Session teardown until all tasks accessing managed code have
  /// completed.
  ///
  /// Clients should prefer using callManagedCode to automatically acquire
  /// and hold a token for the duration of a call.
  TaskGroup::TokenSource managedCodeTokenSource() const {
    return ManagedCodeTaskGroup;
  }

  /// Call managed code.
  ///
  /// This helper tries to acquire a ManagedCodeTaskGroup token and, if
  /// successful, calls the given function object with the given arguments
  /// while holding the token.
  ///
  /// The token is held only for the duration of the (synchronous) call to Fn,
  /// and released as soon as Fn returns; anything Fn runs inline on this thread
  /// before returning is covered by it. Work Fn defers past its return --
  /// stashed to run later, or handed to another thread -- is NOT covered: it
  /// runs on a stack the token no longer guards. Whoever runs such deferred
  /// work is responsible for ensuring a token covers it, typically by acquiring
  /// one (e.g. from a TokenSource, see managedCodeTokenSource) and aborting the
  /// work if the acquire is denied, exactly as for any entry into managed code.
  /// (A resumed continuation cannot bracket its own entry, since its landing
  /// point may itself be managed code.)
  ///
  /// If the token is successfully acquired then this function returns the
  /// result of the call to Fn as a std::optional<T> (for a non-void return
  /// type T), or boolean true (for void returns). Note that for asynchronous
  /// functions (which typically return void) this reflects only that Fn was
  /// invoked, not the result of the asynchronous operation.
  ///
  /// If the token is not successfully acquired then Fn is not called and this
  /// function returns std::nullopt (for a non-void return type) or boolean
  /// false (for void returns).
  ///
  /// See the "Managed code execution and shutdown" section of docs/Design.md
  /// for the model behind managed-code tokens and shutdown.
  template <typename FnT, typename... ArgTs>
  decltype(auto) callManagedCode(FnT &&Fn, ArgTs &&...Args) {
    return ManagedCodeCaller<std::invoke_result_t<FnT, ArgTs...>>::call(
        TaskGroup::Token(ManagedCodeTaskGroup), std::forward<FnT>(Fn),
        std::forward<ArgTs>(Args)...);
  }

  /// Call a tagged handler in the Controller.
  ///
  /// This method can be called directly, but is expected to be more commonly
  /// called via WrapperFunction::call using a CallViaSession object (returned
  /// by the callViaSession method).
  void callController(OnControllerCallReturnFn OnComplete, HandlerTag T,
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

    void operator()(OnControllerCallReturnFn &&HandleResult,
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

  void appendService(std::unique_ptr<Service> Srv);

  /// Attach the given ControllerAccess, using the given BootstrapInfo.
  ///
  /// Upon first call, assuming that the Session has not already been detached
  /// or shutdown, this takes (shared) ownership of CA and calls its connect
  /// method. If detach or shutdown have already been called then this method
  /// will not take ownership of CA or call its connect method.
  ///
  /// This is an implementation detail of the public attach / tryAttach
  /// templates, which are responsible for constructing the ControllerAccess
  /// object: clients never hold a ControllerAccess directly.
  void doAttach(std::shared_ptr<ControllerAccess> CA, BootstrapInfo BI);

  void handleDisconnect();
  void proceedToDetach(std::unique_lock<std::mutex> &Lock,
                       std::shared_ptr<ControllerAccess> TmpCA);
  void detachServices(std::vector<Service *> ToNotify, bool ShutdownRequested);
  void completeDetach();

  void waitForManagedCodeTasksThenShutdown();
  void proceedToShutdown();
  void shutdownServices(std::vector<Service *> ToNotify);
  void completeShutdown();

  void handleWrapperCall(uint64_t CallId, orc_rt_WrapperFunction Fn,
                         WrapperFunctionBuffer ArgBytes) {
    if (!ManagedCodeTaskGroup->acquireToken()) {
      // The ManagedCodeTaskGroup is only closed after detach, so if token
      // acquisition fails we don't try to return an error: the controller
      // should already have signalled error to the caller, and we have no
      // way to transmit an error anyway.
      return;
    }

    RunCall(wrap(this), CallId, &wrapperReturn, Fn, std::move(ArgBytes));
  }

  void sendWrapperResult(uint64_t CallId, WrapperFunctionBuffer ResultBytes);
  static void wrapperReturn(orc_rt_SessionRef S, uint64_t CallId,
                            orc_rt_WrapperFunctionBuffer ResultBytes);

  ExecutorProcessInfo EPI;
  RunWrapperCall RunCall;
  std::shared_ptr<TaskGroup> ManagedCodeTaskGroup = TaskGroup::Create();
  std::shared_ptr<ControllerAccess> CA;
  ErrorReporterFn ReportError;

  mutable std::mutex M;
  std::condition_variable CV;
  State CurrentState = State::Start;
  State TargetState = State::None;
  std::vector<std::unique_ptr<Service>> Services;
  NotificationService &Notifiers;
};

/// Helper function object to report errors via the given Session's
/// reportError method.
class ReportErrorsViaSession {
public:
  explicit ReportErrorsViaSession(Session &S) : S(S) {}
  void operator()(Error Err) const { S.reportError(std::move(Err)); }

private:
  Session &S;
};

} // namespace orc_rt

#endif // ORC_RT_SESSION_H
