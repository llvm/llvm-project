// mod.a.swift
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

public struct S {
  public var a = 1
  fileprivate var s1 = "i"
  fileprivate var s2 = "am"
  fileprivate var s3 = "large"
  fileprivate var s4 = "!"

  public init() {}
}

fileprivate class Message { fileprivate var s = "hello" }
fileprivate struct NotBitwiseTakable {
  fileprivate weak var msg : Message?
}

fileprivate struct FixedContainer {
    var s = S()
}

fileprivate var g_msg = Message()
fileprivate var g_b = NotBitwiseTakable()
fileprivate var g_s = S()
fileprivate var g_t = (S(), S())
fileprivate var g_c = FixedContainer()

public func initGlobal() -> Int {
  g_b.msg = g_msg
  return g_s.a + g_t.0.a + g_c.s.a
}

public func fA(_ x: S) -> Int {
  return x.a
}
