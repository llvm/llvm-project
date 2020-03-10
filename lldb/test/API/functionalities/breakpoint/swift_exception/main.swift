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
enum EnumError : Error
{
    case ImportantError
    case TrivialError
}

class ClassError : Error
{
    var _code : Int = 10
    var _domain : String = "ClassError"
    var m_message : String

    init (_ message: String)
    {
        m_message = message
    }
    func SomeMethod (_ input : Int) 
    {
        print (m_message)  // Set a breakpoint here to test method contexts
    }
}

func IThrowEnumOver10(_ input : Int) throws -> Int
{
    if input > 100
    {
        throw EnumError.ImportantError
    }
    else if input > 10
    {
        throw EnumError.TrivialError
    }
    else
    {
        return input + 2
    }
}

func IThrowObjectOver10(_ input : Int) throws -> Int
{
    if input > 100
    {
        let my_error = ClassError("Over 100")
        throw  my_error
    }
    else if input > 10
    {
        let my_error = ClassError("Over 10 but less than 100")
        throw my_error
    }
    else
    {
        return input + 2
    }
}

do
{
    try IThrowEnumOver10(101)  // Set a breakpoint here to run expressions
    try IThrowObjectOver10(11)
}
catch (let e)
{
    print (e, true)
}
