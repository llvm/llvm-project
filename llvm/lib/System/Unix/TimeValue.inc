//===- Unix/TimeValue.cpp - Unix TimeValue Implementation -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Unix specific portion of the TimeValue class.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

#include "Unix.h"

namespace llvm {
  using namespace sys;

std::string TimeValue::toString() const {
  char buffer[32];

  time_t ourTime = time_t(this->toEpochTime());
#ifdef __hpux
// note that the following line needs -D_REENTRANT on HP-UX to be picked up 
  asctime_r(localtime(&ourTime), buffer);
#else
  ::asctime_r(::localtime(&ourTime), buffer);
#endif

  std::string result(buffer);
  return result.substr(0,24);
}

TimeValue TimeValue::now() {
  struct timeval the_time;
  timerclear(&the_time);
  if (0 != ::gettimeofday(&the_time,0)) {
    // This is *really* unlikely to occur because the only gettimeofday
    // errors concern the timezone parameter which we're passing in as 0.
    // In the unlikely case it does happen, just return MinTime, no error
    // message needed. 
    return MinTime;
  }

  return TimeValue(
    static_cast<TimeValue::SecondsType>( the_time.tv_sec ), 
    static_cast<TimeValue::NanoSecondsType>( the_time.tv_usec * 
      NANOSECONDS_PER_MICROSECOND ) );
}

}
