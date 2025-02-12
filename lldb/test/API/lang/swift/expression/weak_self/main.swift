// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2019 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------

class ClosureMaker {
  var a : Int

  init (a : Int) {
    self.a = a
  }

  func getClosure() -> (() -> Int) {
    return { [weak self] () -> Int in
             if let self = self {
               return self.a // break here for if let success
             } else {
               return 0 // break here for if let else
            }
          }
  }

  func getGuardClosure() -> (() -> Int) {
    return { [weak self] () -> Int in
             guard let self = self else {
               return 0 // break here for guard let else
             }
             return self.a // break here for guard let success
          }
  }
}

func use<T>(_ t: T) {}

var livemaker : ClosureMaker? = ClosureMaker(a: 5)
let liveclosure = livemaker!.getClosure()
let liveguardclosure = livemaker!.getGuardClosure()
use((liveclosure(), liveguardclosure()))

var deadmaker : ClosureMaker? = ClosureMaker(a: 3)
let deadclosure = deadmaker!.getClosure()
let deadguardclosure = deadmaker!.getGuardClosure()
deadmaker = nil
use((deadclosure(), deadguardclosure()))
