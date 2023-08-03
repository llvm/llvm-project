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
class Foo {
    var _a : Int
    var _b : Int
    init ()
    {
        _a = 22
        _b = 33
    }
}

struct SmallStruct
{
    var _a : Int
    var _b : Int
    var _c : Int
    var _d : Int
    init ()
    {
      _a = 22
      _b = 33
      _c = 44
      _d = 55
    }
}

struct StructOfClasses
{
  var _a : Foo
  var _b : Foo
  var _c : Foo
  init()
  {
    _a = Foo()
    _b = Foo()
    _c = Foo()
  }
}

struct BigStruct
{
   var _a : SmallStruct
   var _b : SmallStruct
   init ()
   {
       _a = SmallStruct()
       _b = SmallStruct()
   }
}

enum MyError : Error
{
    case TrivialError
    case WorrisomeError
    case SeriousError
}

func returnUInt64() -> UInt64 {
    return UInt64(123) // Set breakpoint here
}
func returnInt() -> Int64 {
    return -123 // Set breakpoint here
}
func returnFloat() -> Float {
    return Float(1.25) // Set float breakpoint here
}
func returnDouble() -> Double {
    return 2.125 // Set float breakpoint here
}
func returnClass () -> Foo {
    return Foo() // Set breakpoint here
}

func returnSmallStruct() -> SmallStruct
{
    return SmallStruct() // Set breakpoint here
}

func returnBigStruct() -> BigStruct
{
    return BigStruct() // Set another breakpoint here
}

func returnStructOfClasses() -> StructOfClasses
{
    return StructOfClasses() // Set breakpoint here
}

func returnString() -> String {
    return "Hello World" // Set breakpoint here
}
func getDict() -> Dictionary<Int, String>
{
    let d = Dictionary<Int, String>()
    return d // Set breakpoint here
}
func getOptionalString() -> String?
{
    let opt_str = Optional<String>.some("Hello")
    return opt_str // Set breakpoint here
}

func throwAnError(_ should_throw : Bool) throws -> Int
{
    if should_throw  // Set breakpoint here
    {
        throw MyError.SeriousError 
    }
    return 10
}

func main() -> Int {
    let u = returnUInt64()
    let i = returnInt()
    let c = returnClass()
    let ss = returnSmallStruct()
    let cs = returnStructOfClasses()
    let s = returnString()
    let dict = getDict()
    let opt_str = getOptionalString()
    let f = returnFloat()
    let d = returnDouble()
    do
    {
        let not_err = try throwAnError(false)
        let not_set = try throwAnError(true)
        print ("\(not_err) \(not_set)")
    }
    catch (let err)
    {
        print (err)
    }

    let bs = returnBigStruct()

    // TODO: remove the line below when it is no longer needed. Currently extra
    // line table entries will be added after "returnDouble()" and will stop the
    // test from working
    print("\(u) \(i) \(c) \(s) \(dict) \(opt_str) \(f) \(d)")
    return 0
}

main()
