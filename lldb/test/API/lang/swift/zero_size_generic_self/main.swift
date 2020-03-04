// main.swift
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2019 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// -----------------------------------------------------------------------------

func use<T>(_ t : T) {}

struct GenericSelf<T> {
  init(x: T) {
    use(x) //%self.expect('frame variable -d run -- self', substrs=['GenericSelf<String>'])
  }
}

// More complex example with protocol extensions.

protocol MyKey {}
extension Int : MyKey {}
protocol MyProtocol {
  associatedtype Key : MyKey
}
struct MyStruct<S : MyKey> : MyProtocol {
  typealias Key = S
}
extension MyProtocol {
    func decode() {
        use(self) //%self.expect('frame variable -d run -- self', substrs=['(a.MyStruct<Int>)', 'self', '=', '{}'])
        return
    }
}

// Run.
GenericSelf(x: "hello world")
let s = MyStruct<Int>()
s.decode()
