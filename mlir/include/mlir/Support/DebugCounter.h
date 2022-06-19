//===- DebugCounter.h - Debug Counter support -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_DEBUGCOUNTER_H
#define MLIR_SUPPORT_DEBUGCOUNTER_H

#include "mlir/Support/DebugAction.h"
#include "llvm/ADT/StringMap.h"
#include <string>

namespace mlir {

/// This class implements a debug action handler that attaches a counter value
/// to debug actions and enables/disables execution of these action based on the
/// value of the counter. The counter controls the execution of the action with
/// a "skip" and "count" value. The "skip" value is used to skip a certain
/// number of initial executions of a debug action. The "count" value is used to
/// prevent a debug action from executing after it has executed for a set number
/// of times (not including any executions that have been skipped). For example,
/// a counter for a debug action with `skip=47` and `count=2`, would skip the
/// first 47 executions, then execute twice, and finally prevent any further
/// executions.
class DebugCounter : public DebugActionManager::GenericHandler {
public:
  DebugCounter();
  ~DebugCounter() override;

  /// Add a counter for the given debug action tag. `countToSkip` is the number
  /// of counter executions to skip before enabling execution of the action.
  /// `countToStopAfter` is the number of executions of the counter to allow
  /// before preventing the action from executing any more.
  void addCounter(StringRef actionTag, int64_t countToSkip,
                  int64_t countToStopAfter);

  /// Register a counter with the specified name.
  FailureOr<bool> shouldExecute(StringRef tag, StringRef description) final;

  /// Print the counters that have been registered with this instance to the
  /// provided output stream.
  void print(raw_ostream &os) const;

  /// Register the command line options for debug counters.
  static void registerCLOptions();

private:
  /// Apply the registered CL options to this debug counter instance.
  void applyCLOptions();

  /// This struct represents a specific counter being tracked.
  struct Counter {
    Counter(int64_t countToSkip = 0, int64_t countToStopAfter = -1)
        : countToSkip(countToSkip), countToStopAfter(countToStopAfter) {}

    /// The current count of this counter.
    int64_t count{0};
    /// The number of initial executions of this counter to skip.
    int64_t countToSkip;
    /// The number of times to execute this counter before stopping.
    int64_t countToStopAfter;
  };

  /// A mapping between a given action tag and its counter information.
  llvm::StringMap<Counter> counters;
};

} // namespace mlir

#endif
