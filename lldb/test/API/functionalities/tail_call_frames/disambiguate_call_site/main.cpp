//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

volatile int x;

void __attribute__((noinline)) sink() {
  x++; //% self.filecheck("bt", "main.cpp", "-implicit-check-not=artificial")
  // CHECK: frame #0: 0x{{[0-9a-f]+}} a.out`sink() at main.cpp:[[@LINE-1]]:4 [opt]
  // CHECK-NEXT: func2{{.*}} [opt] [artificial]
  // CHECK-NEXT: main{{.*}} [opt]
}

void __attribute__((noinline)) func2() {
  sink(); /* tail */
}

void __attribute__((noinline)) func1() { sink(); /* tail */ }

int __attribute__((disable_tail_calls)) main(int argc, char **) {
  // The sequences `main -> f{1,2} -> sink` are both plausible. Test that
  // return-pc call site info allows lldb to pick the correct sequence.
  func2();
  if (argc == 100)
    func1();
  return 0;
}
