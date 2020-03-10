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

func main() -> Int {
    let short_five : Int8 = 5
    let short_four : Int8 = 4

    var reg_true : Bool   = true
    var reg_false : Bool  = false
    var odd_true : Bool   = unsafeBitCast(short_five, to: Bool.self)
    var odd_false : Bool  = unsafeBitCast(short_four, to: Bool.self)

    var odd_true_works = reg_true == odd_true
    var odd_false_works = reg_false == odd_false

    print("stop here") // Set breakpoint here
    return 0
}

main()
