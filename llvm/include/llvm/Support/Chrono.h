//===- llvm/Support/Chrono.h - Utilities for Timing Manipulation-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CHRONO_H
#define LLVM_SUPPORT_CHRONO_H

#include "llvm/Support/Compiler.h"

#include <chrono>
#include <ctime>

namespace llvm {

class raw_ostream;

namespace sys {

/// A time point on the system clock. This is provided for two reasons:
/// - to insulate us agains subtle differences in behavoir to differences in
///   system clock precision (which is implementation-defined and differs between
///   platforms).
/// - to shorten the type name
/// The default precision is nanoseconds. If need a specific precision specify
/// it explicitly. If unsure, use the default. If you need a time point on a
/// clock other than the system_clock, use std::chrono directly.
template <typename D = std::chrono::nanoseconds>
using TimePoint = std::chrono::time_point<std::chrono::system_clock, D>;

/// Convert a TimePoint to std::time_t
LLVM_ATTRIBUTE_ALWAYS_INLINE inline std::time_t toTimeT(TimePoint<> TP) {
  using namespace std::chrono;
  return system_clock::to_time_t(
      time_point_cast<system_clock::time_point::duration>(TP));
}

/// Convert a std::time_t to a TimePoint
LLVM_ATTRIBUTE_ALWAYS_INLINE inline TimePoint<std::chrono::seconds>
toTimePoint(std::time_t T) {
  using namespace std::chrono;
  return time_point_cast<seconds>(system_clock::from_time_t(T));
}

} // namespace sys

raw_ostream &operator<<(raw_ostream &OS, sys::TimePoint<> TP);

} // namespace llvm

#endif // LLVM_SUPPORT_CHRONO_H
