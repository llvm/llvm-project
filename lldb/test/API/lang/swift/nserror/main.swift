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

import Foundation

func someAPI(_ x : Int, _ y : Int) -> (String, NSError) {
  if (x == 0 && y == 0) {
    return (" ",NSError(domain: "lldbrocks",code: 0xBABEFEED, userInfo: ["x":x,"y":y]))
  } else {
    return ("x+y",NSError(domain: "lldbrocks", code: 0, userInfo: ["x":x,"y":y]))
  }
}

func main() {
  var call1 = someAPI(0,0)
  var call2 = someAPI(3,4)
  print("// Set a breakpoint here")
}

main()

