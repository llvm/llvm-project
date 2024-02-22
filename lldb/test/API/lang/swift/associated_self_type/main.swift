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
extension Collection {
  func foo(_ x: Iterator.Element) {
    print(x) //%self.expect('frame variable -d run -- x', substrs=['key = 2', 'value = 2'])
  }
}

func main() {
  var a = [2: 2]
  for element in a {
    a.foo((key: element.key, value: element.value))
  }
}

main()
