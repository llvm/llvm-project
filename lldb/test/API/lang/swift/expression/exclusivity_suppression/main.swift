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

struct S {
  var i = 7

  func get() -> Int {
    // Breakpoint 2
    return i
  }

  // Mutating function inc will acquire write access to i.
  mutating func inc() {
    i += 1

    // Breakpoint 1
  }
}

// We wrap our test class C so that we're reading a member variable, not a
// global. LLDB accesses globals via unsafe pointers, which aren't subject to
// dynamic exclusivity checks, and we need to actually hit these checks to
// verify that enforcement is suppressed.
class Wrapper {
  var s = S()
}

let w = Wrapper()
w.s.inc()
