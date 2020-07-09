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
import mod

// This is the same as mod.S to make it easier to
// compare the reconstruction of the resilient and non-resilient types.
struct T {
  public var a = 2
  fileprivate var s1 = "I"
  fileprivate var s2 = "AM"
  fileprivate var s3 = "LARGE"
  fileprivate var s4 = "!!"

  public init() {}
}

func main() {
  initGlobal()
  let s = S()
  let s_opt : Optional<S> = s
  let t = T()
  let t_opt : Optional<T> = t
  print(s.a) // break here
}

main()

