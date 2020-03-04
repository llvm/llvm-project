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
class Minion {
  var x = 1
  
  static var TheMinionKing = Minion()
}

class MyClass {
    /*default*/     var sub_001: Minion! = Minion.TheMinionKing
    weak            var sub_002: Minion! = Minion.TheMinionKing
    unowned         var sub_003: Minion! = Minion.TheMinionKing
    unowned(unsafe) var sub_004: Minion! = Minion.TheMinionKing
    unowned         var sub_005: Minion  = Minion.TheMinionKing
    unowned(unsafe) var sub_006: Minion  = Minion.TheMinionKing
}

func main() {
  var myclass = MyClass()
  print(myclass) // Set breakpoint here
}

main()
