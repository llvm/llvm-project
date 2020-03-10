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
public class Base { let name = "hardcodedstring" }
public class Derived1: Base {}
public class Derived2: Derived1 {}

func testDisappearingStringMember() 
{
    print("")
    var base = Base()
    var derived1 = Derived1()
    var derived2 = Derived2()
    let thename = derived2.name
    print("--- break here ---") //% self.expect("frame variable base", substrs=['"hardcodedstring"'])
    //% self.expect("frame variable derived1", substrs=['"hardcodedstring"'])
    //% self.expect("frame variable derived2", substrs=['"hardcodedstring"'])
}

testDisappearingStringMember()
