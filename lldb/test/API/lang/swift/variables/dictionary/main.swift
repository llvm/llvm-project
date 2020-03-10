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
class Wrapper {
  let value: String
  
  init(value: String) {
    self.value = value
  }
}

func main() {
  var d = [Int: Wrapper]()
  for i in 0..<100 {
    d[i] = Wrapper(value: "\(i * 2 + 1)")
  }
  print(d) // break here
}

main()
