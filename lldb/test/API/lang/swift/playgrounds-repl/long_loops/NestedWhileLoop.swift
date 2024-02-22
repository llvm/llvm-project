// WhileLoop.swift
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

//: Playground - noun: a place where people can play

x = 0
var y = 0
while x < 10 {
    while y < 1000 {
        let z = x + y
        y += 1
    }
    x += 1
}
