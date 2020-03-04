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
class Base {
    var b : String
    init () {
        b = "Hello"
    }
}
class Foo : Base {
    var x : Int
    var y : Float
    override init () {
        x = 12
        y = 2.25
        super.init()
    }
}

func main() -> Int
{
    var f = Foo()
    return f.x// Set breakpoint here
}

main()
