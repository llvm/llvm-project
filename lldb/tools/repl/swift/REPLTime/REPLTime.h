// REPLTime.h
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2017 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------

#ifndef REPL_TIME_H
#define REPL_TIME_H

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;
typedef int kern_return_t;

struct mach_timebase_info {
  uint32_t numer;
  uint32_t denom;
};

typedef struct mach_timebase_info *mach_timebase_info_t;

extern kern_return_t mach_timebase_info(mach_timebase_info_t info);
extern uint64_t mach_absolute_time();

kern_return_t __attribute__ ((always_inline)) REPL_mach_timebase_info(mach_timebase_info_t info) {
  return mach_timebase_info(info);
}

uint64_t __attribute__ ((always_inline)) REPL_mach_absolute_time() {
  return mach_absolute_time();
}

#endif //REPL_TIME_H