//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class A {
public:
    virtual int foo() { return 1; }
    virtual ~A () = default;
    A() = default;
};

class B : public A {
public:
    virtual int foo() { return 2; }
    virtual ~B () = default;
    B() = default;
};

int main() {
    A* a = new B();
    a->foo();  // break here
    return 0;  // break here
}

