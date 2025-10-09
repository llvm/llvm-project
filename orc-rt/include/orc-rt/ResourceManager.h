//===- ResourceManager.h -- Interface for JIT resource managers -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ResourceManager class and related APIs.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_RESOURCEMANAGER_H
#define ORC_RT_RESOURCEMANAGER_H

#include "orc-rt/Error.h"
#include "orc-rt/move_only_function.h"

namespace orc_rt {

/// A ResourceManager manages resources (e.g. JIT'd memory) to support a JIT
/// session.
class ResourceManager {
public:
  using OnCompleteFn = move_only_function<void(Error)>;

  virtual ~ResourceManager();

  /// The detach method will be called if the controller disconnects from the
  /// session without shutting the session down.
  ///
  /// Since no further requests for allocation will be made, the ResourceManager
  /// may discard any book-keeping data-structures used to support allocation.
  /// E.g. a JIT memory manager may discard its free-list, since no further
  /// JIT'd allocations will happen.
  virtual void detach(OnCompleteFn OnComplete) = 0;

  /// The shutdown operation will be called at the end of the session.
  /// The ResourceManager should release all held resources.
  virtual void shutdown(OnCompleteFn OnComplete) = 0;
};
} // namespace orc_rt

#endif // ORC_RT_RESOURCEMANAGER_H
