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
class C {
  private struct Nested { public static var g_counter = 1 }
  let m_counter: Int
  private init(_ val: Int) { m_counter = val }
  class func Create() -> C {
    Nested.g_counter += 1
    return C(Nested.g_counter)
  }
}

func main() {
  var aInt = [[1,2,3,4,5],[1,2,3,4],[1,2,3],[1,2],[1],[]]
  var aC = [[C.Create(),C.Create(),C.Create(),C.Create()],[C.Create(),C.Create()],[],[C.Create()],[C.Create(),C.Create()]]
  print(0) // break here
}

main()

