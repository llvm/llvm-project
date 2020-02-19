//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

volatile int x;

void __attribute__((noinline)) tail_call_sink() {
  x++; //% self.filecheck("bt", "main.cpp", "-check-prefix=TAIL-CALL-SINK")
  // TAIL-CALL-SINK: frame #0: 0x{{[0-9a-f]+}} a.out`tail_call_sink() at main.cpp:[[@LINE-1]]:4 [opt]
  // TAIL-CALL-SINK-NEXT: func3{{.*}} [opt] [artificial]
  // TAIL-CALL-SINK-NEXT: main{{.*}} [opt]

  // TODO: The backtrace should include inlinable_function_which_tail_calls.
}

void __attribute__((always_inline)) inlinable_function_which_tail_calls() {
  tail_call_sink();
}

void __attribute__((noinline)) func3() {
  inlinable_function_which_tail_calls();
}

void __attribute__((always_inline)) inline_sink() {
  x++; //% self.filecheck("bt", "main.cpp", "-check-prefix=INLINE-SINK")
  // INLINE-SINK: frame #0: 0x{{[0-9a-f]+}} a.out`func2() [inlined] inline_sink() at main.cpp:[[@LINE-1]]:4 [opt]
  // INLINE-SINK-NEXT: func2{{.*}} [opt]
  // INLINE-SINK-NEXT: func1{{.*}} [opt] [artificial]
  // INLINE-SINK-NEXT: main{{.*}} [opt]
}

void __attribute__((noinline)) func2() { inline_sink(); /* inlined */ }

void __attribute__((noinline)) func1() { func2(); /* tail */ }

int __attribute__((disable_tail_calls)) main() {
  // First, call a function that tail-calls a function, which itself inlines
  // a third function.
  func1();

  // Next, call a function which contains an inlined tail-call.
  func3();

  return 0;
}
