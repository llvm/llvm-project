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

  /// The onDetach method will be called if the controller disconnects from the
  /// session without shutting the session down.
  ///
  /// Since no further requests to the Service will be made, the Service may
  /// discard any book-keeping data-structures that are only needed to serve
  /// ongoing requests. E.g. a JIT memory manager may discard its free-list,
  /// since no further JIT'd allocations will happen.
  virtual void onDetach(OnCompleteFn OnComplete) = 0;

  /// The onShutdown operation will be called at the end of the session.
  ///
  /// The Service should release any held resources.
  virtual void onShutdown(OnCompleteFn OnComplete) = 0;
};
} // namespace orc_rt

#endif // ORC_RT_SERVICE_H
