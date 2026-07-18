//===-------- Service.h - Interface for Session Services --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Service class and related APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SERVICE_H
#define ORC_RT_SERVICE_H

#include "orc-rt/Error.h"
#include "orc-rt/move_only_function.h"

namespace orc_rt {

/// A Service typically manages some resource(s) or performs some actions on
/// behalf of a Session. E.g. a Memory Manager service.
/// Services are owned by the Session and notified when the controller
/// detaches, and when the Session shuts down.
class Service {
public:
  using OnCompleteFn = move_only_function<void()>;

  virtual ~Service();

  /// Called when controller access becomes permanently unavailable.
  ///
  /// This is guaranteed to be called exactly once before onShutdown,
  /// regardless of how the Session reaches that state. It is called when:
  ///   - The controller explicitly disconnects.
  ///   - The Session::detach() method is called.
  ///   - The Session shuts down (even if no controller was ever attached).
  ///
  /// After onDetach is called no further requests will be made to the Service
  /// by the controller. Note that JIT'd code may continue to make requests to
  /// the service concurrent with a call to onDetach.
  ///
  /// If ShutdownRequested is true then a Session shutdown is already pending,
  /// and will proceed after all Services have been notified of the detach.
  ///
  /// onDetach provides an opportunity for Services to release any resources
  /// that are only required while the Session is attached to the controller.
  /// It is expected that many Services will implement this operation as a
  /// no-op.
  virtual void onDetach(OnCompleteFn OnComplete, bool ShutdownRequested) = 0;

  /// The onShutdown operation will be called at the end of the session.
  ///
  /// The Service should release any held resources.
  virtual void onShutdown(OnCompleteFn OnComplete) = 0;
};
} // namespace orc_rt

#endif // ORC_RT_SERVICE_H
