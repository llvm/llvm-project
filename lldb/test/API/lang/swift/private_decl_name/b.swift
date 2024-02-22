// b.swift
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
extension S {
  private class A {
    var a = 3
    var b = "goodbye"
    var c = 1.25
  }
  
  fileprivate func fA() -> Int {
    var a = A()
    return a.a + 1 // break here
  }
}

public func g(_ x: S) -> Int {
  return 2 + x.fA()
}
