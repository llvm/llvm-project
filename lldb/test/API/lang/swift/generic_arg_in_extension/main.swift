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
  func group<Key: Hashable>(f: Key) -> Key {
    print(f) //%self.expect('frame variable -d run -- f', substrs=['= 123456'])
    return f
  }
}

func main() {
  var a = [1,2,3,4,5]
  print(a.group(f: 123456))
}

main()
