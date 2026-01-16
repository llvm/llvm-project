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

#include "orc-rt/Error.h"
#include "orc-rt/ResourceManager.h"
#include "orc-rt/TaskDispatcher.h"
#include "orc-rt/WrapperFunction.h"
#include "orc-rt/move_only_function.h"

#include "orc-rt-c/CoreTypes.h"
#include "orc-rt-c/WrapperFunction.h"

#include <cassert>
#include <condition_variable>
#include <memory>
#include <mutex>
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
public:
  using ErrorReporterFn = move_only_function<void(Error)>;
  using OnShutdownCompleteFn = move_only_function<void()>;

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

    ControllerAccess(Session &S) : S(&S) {}

    /// Called by the Session to disconnect the session with the Controller.
    ///
    /// disconnect implementations must support concurrent entry on multiple
    /// threads, and all calls must block until the disconnect operation is
    /// complete.
    ///
    /// Once disconnect completes, implementations should make no further
    /// calls to the Session, and should ignore any calls from the session
    /// (implementations are free to ignore any calls from the Session after
    /// disconnect is called).
    virtual void disconnect() = 0;

    /// Report an error to the session.
    void reportError(Error Err) {
      assert(S && "Already disconnected");
      S->reportError(std::move(Err));
    }

    /// Call the handler in the controller associated with the given tag.
    virtual void callController(OnCallHandlerCompleteFn OnComplete,
                                HandlerTag T,
                                WrapperFunctionBuffer ArgBytes) = 0;

    /// Send the result of the given wrapper function call to the controller.
    virtual void sendWrapperResult(uint64_t CallId,
                                   WrapperFunctionBuffer ResultBytes) = 0;

    /// Ask the Session to run the given wrapper function.
    ///
    /// Subclasses must not call this method after disconnect returns.
    void handleWrapperCall(uint64_t CallId, orc_rt_WrapperFunction Fn,
                           WrapperFunctionBuffer ArgBytes) {
      assert(S && "Already disconnected");
      S->handleWrapperCall(CallId, Fn, std::move(ArgBytes));
    }

  private:
    void doDisconnect() {
      disconnect();
      S = nullptr;
    }
    Session *S;
  };

  /// Create a session object. The ReportError function will be called to
  /// report errors generated while serving JIT'd code, e.g. if a memory
  /// management request cannot be fulfilled. (Error's within the JIT'd
  /// program are not generally visible to ORC-RT, but can optionally be
  /// reported by calling orc_rc_Session_reportError function.
  ///
  /// Note that entry into the reporter is not synchronized: it may be
  /// called from multiple threads concurrently.
  Session(std::unique_ptr<TaskDispatcher> Dispatcher,
          ErrorReporterFn ReportError)
      : Dispatcher(std::move(Dispatcher)), ReportError(std::move(ReportError)) {
  }

  // Sessions are not copyable or moveable.
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;
  Session(Session &&) = delete;
  Session &operator=(Session &&) = delete;

  ~Session();

  /// Dispatch a task using the Session's TaskDispatcher.
  void dispatch(std::unique_ptr<Task> T) { Dispatcher->dispatch(std::move(T)); }

  /// Report an error via the ErrorReporter function.
  void reportError(Error Err) { ReportError(std::move(Err)); }

  /// Initiate session shutdown.
  ///
  /// Runs shutdown on registered resources in reverse order.
  void shutdown(OnShutdownCompleteFn OnComplete);

  /// Initiate session shutdown and block until complete.
  void waitForShutdown();

  /// Add a ResourceManager to the session.
  void addResourceManager(std::unique_ptr<ResourceManager> RM);

  /// Set the ControllerAccess object.
  void setController(std::shared_ptr<ControllerAccess> CA);

  /// Disconnect the ControllerAccess object.
  void detachFromController();

  void callController(OnCallHandlerCompleteFn OnComplete, HandlerTag T,
                      WrapperFunctionBuffer ArgBytes) {
    if (auto TmpCA = CA)
      CA->callController(std::move(OnComplete), T, std::move(ArgBytes));
    else
      OnComplete(WrapperFunctionBuffer::createOutOfBandError(
          "no controller attached"));
  }

private:
  struct ShutdownInfo {
    bool Complete = false;
    std::condition_variable CompleteCV;
    std::vector<std::unique_ptr<ResourceManager>> ResourceMgrs;
    std::vector<OnShutdownCompleteFn> OnCompletes;
  };

  void shutdownNext(Error Err);
  void shutdownComplete();

  void handleWrapperCall(uint64_t CallId, orc_rt_WrapperFunction Fn,
                         WrapperFunctionBuffer ArgBytes) {
    dispatch(makeGenericTask([=, ArgBytes = std::move(ArgBytes)]() mutable {
      Fn(wrap(this), CallId, wrapperReturn, ArgBytes.release());
    }));
  }

  void sendWrapperResult(uint64_t CallId, WrapperFunctionBuffer ResultBytes) {
    if (auto TmpCA = CA)
      TmpCA->sendWrapperResult(CallId, std::move(ResultBytes));
  }

  static void wrapperReturn(orc_rt_SessionRef S, uint64_t CallId,
                            orc_rt_WrapperFunctionBuffer ResultBytes);

  std::unique_ptr<TaskDispatcher> Dispatcher;
  std::shared_ptr<ControllerAccess> CA;
  ErrorReporterFn ReportError;

  std::mutex M;
  std::vector<std::unique_ptr<ResourceManager>> ResourceMgrs;
  std::unique_ptr<ShutdownInfo> SI;
};

class CallViaSession {
public:
  CallViaSession(Session &S, Session::HandlerTag T) : S(S), T(T) {}

  void operator()(Session::OnCallHandlerCompleteFn &&HandleResult,
                  WrapperFunctionBuffer ArgBytes) {
    S.callController(std::move(HandleResult), T, std::move(ArgBytes));
  }

private:
  Session &S;
  Session::HandlerTag T;
};

} // namespace orc_rt

#endif // ORC_RT_SESSION_H
