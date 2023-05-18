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

public class Klass {
    public func doSomething() {}
}

public protocol P {
    static var value: P { get }
    func doSomething()
}

extension Klass : P {
    public static var value: P { Klass() }
}

var trueBoolValue : Bool { true }
var falseBoolValue : Bool { false }

//////////////////
// Simple Tests //
//////////////////

public func copyableValueTest() {
    print("stop here") // Set breakpoint
    let k = Klass()
    k.doSomething()
    let m = consume k // Set breakpoint
    m.doSomething() // Set breakpoint
}

public func copyableVarTest() {
    print("stop here") // Set breakpoint
    var k = Klass()
    k.doSomething()
    print("stop here") // Set breakpoint
    let m = consume k
    m.doSomething()
    k = Klass()     // Set breakpoint
    k.doSomething() // Set breakpoint
    print("stop here")
}

public func addressOnlyValueTest<T : P>(_ x: T) {
    print("stop here") // Set breakpoint
    let k = x
    k.doSomething()
    let m = consume k // Set breakpoint
    m.doSomething() // Set breakpoint
}

public func addressOnlyVarTest<T : P>(_ x: T) {
    print("stop here") // Set breakpoint
    var k = x
    k.doSomething()
    print("stop here") // Set breakpoint
    let m = consume k
    m.doSomething()
    k = x // Set breakpoint
    k.doSomething() // Set breakpoint
}

//////////////////////
// Arg Simple Tests //
//////////////////////

public func copyableValueArgTest(_ k: __owned Klass) {
    print("stop here") // Set breakpoint
    k.doSomething()
    let m = consume k // Set breakpoint
    m.doSomething() // Set breakpoint
}

public func copyableVarArgTest(_ k: inout Klass) {
    print("stop here") // Set breakpoint
    k.doSomething()
    print("stop here") // Set breakpoint
    let m = consume k
    m.doSomething()
    k = Klass()     // Set breakpoint
    k.doSomething() // Set breakpoint
    print("stop here")
}

public func addressOnlyValueArgTest<T : P>(_ k: __owned T) {
    print("stop here") // Set breakpoint
    k.doSomething()
    let m = consume k // Set breakpoint
    m.doSomething() // Set breakpoint
}

public func addressOnlyVarArgTest<T : P>(_ k: inout T, _ x: T) {
    print("stop here") // Set breakpoint
    k.doSomething()
    print("stop here") // Set breakpoint
    let m = consume k
    m.doSomething()
    k = x // Set breakpoint
    k.doSomething() // Set breakpoint
}

////////////////////////////////////
// Conditional Control Flow Tests //
////////////////////////////////////

public func copyableValueCCFTrueTest() {
    let k = Klass() // Set breakpoint
    k.doSomething() // Set breakpoint
    if trueBoolValue {
        let m = consume k // Set breakpoint
        m.doSomething() // Set breakpoint
    }
    // Set breakpoint
}

public func copyableValueCCFFalseTest() {
    let k = Klass() // Set breakpoint
    k.doSomething() // Set breakpoint
    if falseBoolValue {
        let m = consume k
        m.doSomething()
    }
    // Set breakpoint
}

public func copyableVarTestCCFlowTrueReinitOutOfBlockTest() {
    var k = Klass() // Set breakpoint
    k.doSomething()
    if trueBoolValue {
        print("stop here") // Set breakpoint
        let m = consume k
        m.doSomething() // Set breakpoint
    }
    k = Klass() // Set breakpoint
    k.doSomething() // Set breakpoint
}

public func copyableVarTestCCFlowTrueReinitInBlockTest() {
    var k = Klass() // Set breakpoint
    k.doSomething()
    if trueBoolValue {
        print("stop here") // Set breakpoint
        let m = consume k
        m.doSomething()
        k = Klass() // Set breakpoint
        k.doSomething() // Set breakpoint
    }
    k.doSomething() // Set breakpoint
}

public func copyableVarTestCCFlowFalseReinitOutOfBlockTest() {
    var k = Klass() // Set breakpoint
    k.doSomething() // Set breakpoint
    if falseBoolValue {
        let m = consume k
        m.doSomething()
    }
    k = Klass() // Set breakpoint
    k.doSomething() // Set breakpoint
}

public func copyableVarTestCCFlowFalseReinitInBlockTest() {
    var k = Klass() // Set breakpoint
    k.doSomething()  // Set breakpoint
    if falseBoolValue {
        let m = consume k
        m.doSomething()
        k = Klass()
    }
    k.doSomething() // Set breakpoint
}

//////////////////////////
// Top Level Entrypoint //
//////////////////////////

func main() {
    copyableValueTest()
    copyableVarTest()
    addressOnlyValueTest(Klass())
    addressOnlyVarTest(Klass())
    copyableValueArgTest(Klass())
    var kls = Klass()
    copyableVarArgTest(&kls)
    addressOnlyValueArgTest(Klass())
    addressOnlyVarArgTest(&kls, Klass())
    copyableValueCCFTrueTest()
    copyableValueCCFFalseTest()
    copyableVarTestCCFlowTrueReinitOutOfBlockTest()
    copyableVarTestCCFlowTrueReinitInBlockTest()
    copyableVarTestCCFlowFalseReinitOutOfBlockTest()
    copyableVarTestCCFlowFalseReinitInBlockTest()
}

main()
