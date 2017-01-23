// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------

// The following #if can be removed once the fix for rdar://problem/23332517
// lands in an external Xcode build.
#if XCODE_BUILD_ME || !_runtime(_ObjC)
#if _runtime(_ObjC)
import Darwin
#endif

func repl_main() -> Int
{
    return 0
}

#if _runtime(_ObjC)
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

var timebase_info = mach_timebase_info(numer: 0, denom: 0)
Darwin.mach_timebase_info(&timebase_info)
var subsequent_short = 0
while subsequent_short < 3 {
    var start = Darwin.mach_absolute_time()
    repl_main()
    var end = Darwin.mach_absolute_time()
    var elapsedTicks = end - start
    var elapsedNano = (elapsedTicks * UInt64(timebase_info.numer)) / UInt64(timebase_info.denom)
    if elapsedNano < 100000 {
        subsequent_short += 1
    } else {
        subsequent_short = 0
    }
}
#else
repl_main()
#endif
#endif

