; REQUIRES: asserts
; RUN: llc < %s -O0 -mtriple=riscv64 -debug-only=isel 2>&1 | FileCheck %s

define i32* @fooOptnone(i32* %p, i32* %q, i32** %z) #0 {
; CHECK: Changing optimization level for Function fooOptnone
; CHECL: Before: -O2 ; After: -O0

; CHECK: Restoring optimization level for Function fooOptnone
; CHECK: Before: -O0 ; After: -O2
entry:
  %r = load i32, i32* %p
  %s = load i32, i32* %q
  %y = load i32*, i32** %z

  %t0 = add i32 %r, %s
  %t1 = add i32 %t0, 1
  %t2 = getelementptr i32, i32* %y, i32 1
  %t3 = getelementptr i32, i32* %t2, i32 %t1

  ret i32* %t3

}

attributes #0 = { nounwind optnone noinline }
