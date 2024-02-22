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
import Dispatch

let q = DispatchQueue.global()
let g = DispatchGroup()
let s = DispatchSemaphore(value: 0)
var arr = [1, 2, 3]
g.enter()
q.async {
    _ = arr.count
    s.wait()
    g.leave()
}
g.enter()
q.async {
    arr.append(5)
    s.signal()
    g.leave()
}
g.wait()
