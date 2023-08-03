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
struct SError {
	var payload = 1
}

class CError {
	var payload = 2
}

enum EError {
	case SucksToBeYou
	case OutOfCookies
}

extension SError : Error {
	var _domain: String { get { return "" } }
	var _code: Int { get { return 0 } }
}

extension CError : Error {
	var _domain: String { get { return "" } }
	var _code: Int { get { return 0 } }
}

extension EError : Error {
	var _domain: String { get { return "" } }
	var _code: Int { get { return 0 } }
}

func main() {
	var s: Error = SError()
	var c: Error = CError()
	var e: Error = EError.OutOfCookies
	
	print("break here") //% self.expect('frame variable -d run -- s', substrs=['SError','payload = 1'])
	print("break here") //% self.expect('frame variable -d run -- c', substrs=['CError','payload = 2'])
	print("break here") //% self.expect('frame variable -d run -- e', substrs=['EError','OutOfCookies'])

	print("break here") //% self.expect('expr -d run -- s', substrs=['SError','payload = 1'])
	print("break here") //% self.expect('expr -d run -- c', substrs=['CError','payload = 2'])
	print("break here") //% self.expect('expr -d run -- e', substrs=['EError','OutOfCookies'])
}

main()
