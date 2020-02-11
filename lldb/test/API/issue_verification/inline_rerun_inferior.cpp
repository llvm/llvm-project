//===-- main.cpp --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
typedef int Foo;

int main() {
    Foo array[3] = {1,2,3};
    return 0; //% self.expect("frame variable array --show-types --", substrs = ['(Foo [3]) wrong_type_here = {','(Foo) [0] = 1','(Foo) [1] = 2','(Foo) [2] = 3'])
}
