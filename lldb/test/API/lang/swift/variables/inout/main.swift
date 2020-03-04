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
class Class {
    var ivar : Int
    init() 
    { 
        ivar = 1234 
    }
}

class Other : Class {
    var ovar : Int
    override init () 
    { 
        ovar = 112233
        super.init()
        ivar = 4321
    }
    init (in1: Int, in2: Int)
    {
      ovar = in2
      super.init()
      ivar = in1
    }
}

struct Struct {
	var ivar : Int
	init() { ivar = 4567 }
}

func foo (_ x: inout Class) {
	print(x.ivar)
	x.ivar += 1 // Set breakpoint here for Class access
}

func foo(_ x: inout Struct) {
	print(x.ivar)
	x.ivar += 1 // Set breakpoint here for Struct access
}

func fn_ptrs (_ str: Struct) {
  let dict: [Int : (_ str: inout Struct) -> Void] =
      [ 0 : foo]
  let fn = dict[str.ivar] // Set breakpoint here for Function type with inout
  if fn != nil {
    print("Found function")
  }
}

func foo (_ x: inout String) {
  print(x)
  x = "Set breakpoint here for String access"
}

func main() {
  var x : Class = Other()
  var s = Struct()
  var t = "Keep going, nothing to see"

  foo(&x)
  foo(&s)
  foo(&x)
  foo(&t)
  foo(&x)
  foo(&s)

  fn_ptrs(s)
}

main()

