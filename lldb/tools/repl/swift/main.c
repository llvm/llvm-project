// main.c
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2017 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------

#ifdef __APPLE__
#include <mach/mach_time.h>
#include <dlfcn.h>
#endif

#if defined(__linux__)
#include <dlfcn.h>
#endif

#define REPL_MAIN _TF10repl_swift9repl_mainFT_Si

#if !defined(__has_attribute)
#define __has_attribute(attribute) 0
#endif

#if __has_attribute(__optnone__)
#define SWIFT_REPL_NOOPT __attribute__((__optnone__))
#else
#define SWIFT_REPL_NOOPT
#endif

SWIFT_REPL_NOOPT
int REPL_MAIN() {
  return 0;
}

SWIFT_REPL_NOOPT
int main() {
#ifdef __APPLE__
  // Force loading of libswiftCore.dylib, which is not linked at build time.
  dlopen("@rpath/libswiftCore.dylib", RTLD_LAZY);
#elif defined(__linux__)
  dlopen("libswiftCore.so", RTLD_LAZY);
#endif

#ifdef __APPLE__
  // This code will be run when running the REPL. A breakpoint will be set at
  // "repl_main" and we will hit that breakpoint and then allow REPL statements
  // to be evaluated. Since LLDB allows process control, the user can accidentally
  // continue the target:
  //
  // 1> :c
  //
  // So to safeguard against this we hit the breakpoint over and over. If the user
  // detaches:
  //
  // 1> :process detach
  //
  // we want this program to exit without consuming 100% CPU, so we detect any loops
  // that take less than 100us and if we get three of them in a row, we exit.

  struct mach_timebase_info TimebaseInfo;
  mach_timebase_info(&TimebaseInfo);
  int SubsequentShort = 0;
  while (SubsequentShort < 3) {
    const uint64_t Start = mach_absolute_time();
    REPL_MAIN();
    const uint64_t End = mach_absolute_time();
    const uint64_t ElapsedTicks = End - Start;
    const uint64_t ElapsedNano = (ElapsedTicks * (uint64_t)(TimebaseInfo.numer)) / (uint64_t)(TimebaseInfo.denom);
    if (ElapsedNano < 100000) {
      SubsequentShort += 1;
    } else {
      SubsequentShort = 0;
    }
  }
#else
  REPL_MAIN();
#endif
}
