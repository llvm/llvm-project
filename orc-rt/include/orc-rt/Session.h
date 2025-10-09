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
#include "orc-rt/move_only_function.h"

#include <vector>

namespace orc_rt {

/// Represents an ORC executor Session.
class Session {
public:
  using ErrorReporterFn = move_only_function<void(Error)>;
  using OnShutdownCompleteFn = move_only_function<void()>;

  /// Create a session object. The ReportError function will be called to
  /// report errors generated while serving JIT'd code, e.g. if a memory
  /// management request cannot be fulfilled. (Error's within the JIT'd
  /// program are not generally visible to ORC-RT, but can optionally be
  /// reported by calling orc_rc_Session_reportError function.
  ///
  /// Note that entry into the reporter is not synchronized: it may be
  /// called from multiple threads concurrently.
  Session(ErrorReporterFn ReportError) : ReportError(std::move(ReportError)) {}

  // Sessions are not copyable or moveable.
  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  ~Session();

  /// Report an error via the ErrorReporter function.
  void reportError(Error Err) { ReportError(std::move(Err)); }

  /// Initiate session shutdown.
  ///
  /// Runs shutdown on registered resources in reverse order.
  void shutdown(OnShutdownCompleteFn OnComplete);

  /// Initiate session shutdown and block until complete.
  void waitForShutdown();

  /// Add a ResourceManager to the session.
  void addResourceManager(std::unique_ptr<ResourceManager> RM) {
    std::scoped_lock<std::mutex> Lock(M);
    ResourceMgrs.push_back(std::move(RM));
  }

private:
  void shutdownNext(OnShutdownCompleteFn OnShutdownComplete, Error Err,
                    std::vector<std::unique_ptr<ResourceManager>> RemainingRMs);

  std::mutex M;
  ErrorReporterFn ReportError;
  std::vector<std::unique_ptr<ResourceManager>> ResourceMgrs;
};

} // namespace orc_rt

#endif // ORC_RT_SESSION_H
