// one.swift
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
public class Fooey
{
    static var counter = 0

    public init ()
    {
        m_var = 10
    }

    public class func BumpCounter (_ value : Int)
    {
        counter += value
    }
    
    public class func ResetCounter()
    {
        counter = 0
    }

    public class func GetCounter() -> Int
    {
        return counter
    }

    public class func CompareEm1(_ lhs: Fooey, _ rhs : Fooey) -> Bool
    {
        return lhs == rhs     
    }

    public var m_var : Int = 10
}

fileprivate func == (lhs : Fooey, rhs : Fooey) -> Bool 
{ 
    Fooey.BumpCounter(1)
    return lhs.m_var == rhs.m_var
}

