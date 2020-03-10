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

protocol Proto {
	associatedtype FancyType
	func get() -> FancyType
}

extension String : Proto {
	typealias FancyType = Int
	func get() -> Int { return 1 }
}

func function <SomeType : Proto> (_ x : SomeType)
{
	var v = x.get()
	print(v) //Break here
}

class AClass <TypeA>
{
	init () {}
	func method1(_ x : TypeA)
	{
		print("hello world") //Break here
	}
	func method2 <TypeB> (_ x : TypeA, _ y : TypeB)
	{
		print("Hello world") //Break here
	}
}

class OtherClass <TypeA> : AClass <TypeA>
{
	var v : Int
	init(_ x : TypeA) {v = 1234567;super.init()}
}

struct Generic <Foo> {
	var v : Foo
	init (_ _v : Foo) { v = _v }
}

class Pair <Foo,Bar> {
	var one : Foo
	var two : Bar
	
	init (_ o : Foo, _ t : Bar) { one = o; two = t }
	
	func print() {
		Swift.print("hello") //Break here
	}
}

struct Outer <Foo> {
  struct Inner {
    var field: Foo
    
    func print() {
      Swift.print("hello") //Break here
    }
  }
}


func main() {
	var object : AClass<Int> = OtherClass(3)
	object.method1(123)
	function("hello world again")
	var pair1 : Pair<Generic<Int>,Pair<String,Generic<String>>> = 
	 Pair(Generic(193627),Pair("hello",Generic("world")))

	var pair2 = Pair(Generic(3.141592),Generic(Pair("this is","a good thing")))
  pair1.print()
  pair2.print()
	object.method2(5,"hello world")

  let value = Outer<Int>.Inner(field: 10)
  value.print() 
}

main()
