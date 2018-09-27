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

namespace llvm {

class Pass;
class Timer;

/// If -time-passes has been specified, report the timings immediately and then
/// reset the timers to zero.
void reportAndResetTimings();

/// Request the timer for this legacy-pass-manager's pass instance.
Timer *getPassTimer(Pass *);

/// If the user specifies the -time-passes argument on an LLVM tool command line
/// then the value of this boolean will be true, otherwise false.
/// This is the storage for the -time-passes option.
extern bool TimePassesIsEnabled;

} // namespace llvm

#endif
