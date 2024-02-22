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
class A : CustomReflectable {
  var a: A?
  
  var customMirror: Mirror {
    get {
      return Mirror(self, children: ["a" : a], displayStyle: .`class`)
    }
  }
}

struct S {
  var a: A?
}

func main() {
  var s = S()
  s.a = A()
  s.a?.a = A()
  s.a?.a?.a = s.a
  print("a") //%self.expect('po s', substrs=['▿ a : Optional', 'some', '0x', '{ ... }'])
}

main()
