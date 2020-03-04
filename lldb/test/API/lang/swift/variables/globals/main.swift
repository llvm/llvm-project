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
var place_to_stop = 1 // Set top_level breakpoint here
var g_counter = 1

class Foo {
	var x : Int
	init () { x = g_counter; g_counter += 1 }
}

var my_foo = Foo()

struct LargeDude {
	var x : Int
	var y : Int
	var z : Int
	var q : Int
	
	init() { x = 10; y = 20; z = 30; q = 40; }
}

var my_large_dude = LargeDude()

// This line is just to force Swift to realize the globals.
print ("counter: \(g_counter), foo.x: \(my_foo.x), large.y: \(my_large_dude.y)")

func main() {
	print("Hello world") // Set function breakpoint here
}

main()
