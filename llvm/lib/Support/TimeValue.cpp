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
#include "llvm/Config/config.h"

namespace llvm {
using namespace sys;

const TimeValue TimeValue::MinTime       = TimeValue ( INT64_MIN,0 );
const TimeValue TimeValue::MaxTime       = TimeValue ( INT64_MAX,0 );
const TimeValue TimeValue::ZeroTime      = TimeValue ( 0,0 );
const TimeValue TimeValue::PosixZeroTime = TimeValue ( -946684800,0 );
const TimeValue TimeValue::Win32ZeroTime = TimeValue ( -12591158400ULL,0 );

void
TimeValue::normalize( void ) {
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

}

/// Include the platform specific portion of TimeValue class
#ifdef LLVM_ON_UNIX
#include "Unix/TimeValue.inc"
#endif
#ifdef LLVM_ON_WIN32
#include "Windows/TimeValue.inc"
#endif
