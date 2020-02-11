//===-- main.cpp --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
typedef int Foo;

int main() {
    // CHECK: (Foo [3]) array = {
    // CHECK-NEXT: (Foo) [0] = 1
    // CHECK-NEXT: (Foo) [1] = 2
    // CHECK-NEXT: (Foo) [2] = 3
    // CHECK-NEXT: }
    Foo array[3] = {1,2,3};
    return 0; //% self.filecheck("frame variable array --show-types --", 'main.cpp')
}
