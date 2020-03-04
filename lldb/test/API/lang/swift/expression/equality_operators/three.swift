// three.swift
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
import fooey

private var counter = 0

func == (lhs : Fooey, rhs : Fooey) -> Bool
{
    Fooey.BumpCounter(3)
    return lhs.m_var == rhs.m_var + 1  
}

extension Fooey
{
    class func CompareEm3(_ lhs : Fooey, _ rhs : Fooey) -> Bool
    {
        return lhs == rhs
    }
}

var lhs = Fooey()
var rhs = Fooey()

let result1 = Fooey.CompareEm1(lhs, rhs)
Fooey.ResetCounter()
let result2 = Fooey.CompareEm2(lhs, rhs)
Fooey.ResetCounter()
let result3 = Fooey.CompareEm3(lhs, rhs)
