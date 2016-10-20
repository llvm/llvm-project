//===-- TimeValue.cpp - Implement OS TimeValue Concept ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the operating system TimeValue concept.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TimeValue.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/ScopedPrinter.h"

namespace llvm {

using namespace sys;

const TimeValue::SecondsType
  TimeValue::PosixZeroTimeSeconds = -946684800;
const TimeValue::SecondsType
  TimeValue::Win32ZeroTimeSeconds = -12591158400ULL;

void TimeValue::normalize() {
  if ( nanos_ >= NANOSECONDS_PER_SECOND ) {
    do {
      seconds_++;
      nanos_ -= NANOSECONDS_PER_SECOND;
    } while ( nanos_ >= NANOSECONDS_PER_SECOND );
  } else if (nanos_ <= -NANOSECONDS_PER_SECOND ) {
    do {
      seconds_--;
      nanos_ += NANOSECONDS_PER_SECOND;
    } while (nanos_ <= -NANOSECONDS_PER_SECOND);
  }

  if (seconds_ >= 1 && nanos_ < 0) {
    seconds_--;
    nanos_ += NANOSECONDS_PER_SECOND;
  } else if (seconds_ < 0 && nanos_ > 0) {
    seconds_++;
    nanos_ -= NANOSECONDS_PER_SECOND;
  }
}

std::string TimeValue::str() const { return to_string(TimePoint<>(*this)); }

TimeValue TimeValue::now() {
  return TimePoint<>(std::chrono::system_clock::now());
}

} // namespace llvm
