//===- PassTimingInfo.h - pass execution timing -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header defines classes/functions to handle pass execution timing
/// information with interfaces for both pass managers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PASSTIMINGINFO_H
#define LLVM_IR_PASSTIMINGINFO_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Timer.h"
#include <memory>
#include <utility>

namespace llvm {

class Pass;
class PassInstrumentationCallbacks;
class raw_ostream;

/// If -time-passes has been specified, report the timings immediately and then
/// reset the timers to zero. By default it uses the stream created by
/// CreateInfoOutputFile().
void reportAndResetTimings(raw_ostream *OutStream = nullptr);

/// Request the timer for this legacy-pass-manager's pass instance.
Timer *getPassTimer(Pass *);

/// This class implements -time-passes functionality for new pass manager.
/// It provides the pass-instrumentation callbacks that measure the pass
/// execution time. They collect timing info into individual timers as
/// passes are being run. At the end of its life-time it prints the resulting
/// timing report.
class TimePassesHandler {
  /// Value of this type is capable of uniquely identifying pass invocations.
  /// It is a pair of string Pass-Identifier (which for now is common
  /// to all the instance of a given pass) + sequential invocation counter.
  using PassInvocationID = std::pair<StringRef, unsigned>;

  /// A group of all pass-timing timers.
  TimerGroup TG;

  using TimerVector = llvm::SmallVector<std::unique_ptr<Timer>, 4>;
  /// Map of timers for pass invocations
  StringMap<TimerVector> TimingData;

  /// Stack of currently active timers.
  SmallVector<Timer *, 8> TimerStack;

  /// Custom output stream to print timing information into.
  /// By default (== nullptr) we emit time report into the stream created by
  /// CreateInfoOutputFile().
  raw_ostream *OutStream = nullptr;

  bool Enabled;
  bool PerRun;

public:
  TimePassesHandler();
  TimePassesHandler(bool Enabled, bool PerRun = false);

  /// Destructor handles the print action if it has not been handled before.
  ~TimePassesHandler() { print(); }

  /// Prints out timing information and then resets the timers.
  void print();

  // We intend this to be unique per-compilation, thus no copies.
  TimePassesHandler(const TimePassesHandler &) = delete;
  void operator=(const TimePassesHandler &) = delete;

  void registerCallbacks(PassInstrumentationCallbacks &PIC);

  /// Set a custom output stream for subsequent reporting.
  void setOutStream(raw_ostream &OutStream);

private:
  /// Dumps information for running/triggered timers, useful for debugging
  LLVM_DUMP_METHOD void dump() const;

  /// Returns the new timer for each new run of the pass.
  Timer &getPassTimer(StringRef PassID);

  void startTimer(StringRef PassID);
  void stopTimer(StringRef PassID);

  // Implementation of pass instrumentation callbacks.
  void runBeforePass(StringRef PassID);
  void runAfterPass(StringRef PassID);
};

} // namespace llvm

#endif
