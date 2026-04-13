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

var g_url = URL(string: "http://www.apple.com")!

func main() {
  let url = URL(string: "https://www.example.com/path?query#fragment")
  let relativeURL = URL(string: "relative", relativeTo: URL(string: "https://www.example.com/"))

  print("break here")
}

main()
