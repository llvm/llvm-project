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
enum Generic<T> {
  case Case1
  case Case2
  case Case3
}

func foo<T0>(_ x: Generic<T0>) {
  print(x) //% self.expect('frame variable -d run -- x', substrs=['Case1'])
  //% self.expect('frame variable -d run -- x', substrs=['Case2'], matching=False)
  //% self.expect('frame variable -d run -- x', substrs=['Case3'], matching=False)
  //% self.expect('expression -d run -- x', substrs=['Case1'])
  //% self.expect('expression -d run -- x', substrs=['Case2'], matching=False)
  //% self.expect('expression -d run -- x', substrs=['Case3'], matching=False)
}

foo(Generic<Int>.Case1)
