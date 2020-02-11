//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

volatile int x;

void __attribute__((noinline)) sink() {
  x++; //% self.filecheck("bt", "main.cpp")
  // CHECK-NOT: func{{[23]}}
}

void func2();

void __attribute__((noinline)) func1() {
  if (x < 1)
    func2();
  else
    sink();
}

void __attribute__((noinline)) func2() {
  if (x < 1)
    sink();
  else
    func1();
}

int main() {
  // Tail recursion creates ambiguous execution histories.
  x = 0;
  func1();
  return 0;
}
