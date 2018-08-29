//===- PassTimingInfo.h - pass execution timing -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header defines classes/functions to handle pass execution timing
/// information with an interface suitable for both pass managers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PASSTIMINGINFO_H
#define LLVM_IR_PASSTIMINGINFO_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Timer.h"
#include <string>

namespace llvm {

class Pass;
class TimerGroup;

/// Provides a generic interface for collecting pass timing information.
/// Legacy pass managers should specialize with \p PassInfo*.
/// New pass managers should specialize with \p StringRef.
template <typename PassInfoT> class PassTimingInfo {
  StringMap<Timer *> TimingData;
  TimerGroup TG;

public:
  /// Default constructor for yet-inactive timeinfo.
  /// Use \p init() to activate it.
  PassTimingInfo();

  /// Print out timing information and release timers.
  ~PassTimingInfo();

  /// Initializes the static \p TheTimeInfo member to a non-null value when
  /// -time-passes is enabled. Leaves it null otherwise.
  ///
  /// This method may be called multiple times.
  static void init();

  /// Prints out timing information and then resets the timers.
  void print();

  /// Returns the timer for the specified pass if it exists.
  Timer *getPassTimer(PassInfoT);

  static PassTimingInfo *TheTimeInfo;
};

Timer *getPassTimer(Pass *);
Timer *getPassTimer(StringRef);

/// If the user specifies the -time-passes argument on an LLVM tool command line
/// then the value of this boolean will be true, otherwise false.
/// This is the storage for the -time-passes option.
extern bool TimePassesIsEnabled;

} // namespace llvm

#endif
