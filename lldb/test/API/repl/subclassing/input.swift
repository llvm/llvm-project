// input.swift
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

class A {init(a: Int) {}}
///
class B : A {let x: Int; init() { x = 5 + 5; super.init(a: x) } }
///
B().x
///10
extension B : CustomStringConvertible { public var description:String { return "class B\(x) is a subclass of class A"} }
///
B().description
///class B10 is a subclass of class A
