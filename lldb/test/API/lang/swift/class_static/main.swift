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
class WithStatic {
  var a = 1
  var b = 2
  
  static let Shared = WithStatic()
  
  init() {}
}

func main() {
  var v = WithStatic.Shared
  print(1) //% self.expect("frame variable v", substrs=['a = 1', 'b = 2'])
  //% self.expect("expr WithStatic.Shared", substrs=['a = 1', 'b = 2'])
}

main()
