// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2018 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------
import Foundation

// Test that importing Foundation and printing a value for which
// no data formatter exists works consistently on all platforms.
func main() {

  var point = NSMakeRange(23, 42)
  print(point) //% self.expect("frame variable -- point", substrs=['23', '42'])
               //% self.expect("expression -- point", substrs=['23', '42'])
}

main()
