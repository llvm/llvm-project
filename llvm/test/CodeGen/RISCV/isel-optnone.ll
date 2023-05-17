; REQUIRES: asserts
; RUN: llc < %s -O0 -mtriple=riscv64 -debug-only=isel 2>&1 | FileCheck %s

define ptr @fooOptnone(ptr %p, ptr %q, ptr %z) #0 {
; CHECK-NOT: Changing optimization level for Function fooOptnone
; CHECK-NOT: Restoring optimization level for Function fooOptnone

entry:
  %r = load i32, ptr %p
  %s = load i32, ptr %q
  %y = load ptr, ptr %z

  %t0 = add i32 %r, %s
  %t1 = add i32 %t0, 1
  %t2 = getelementptr i32, ptr %y, i32 1
  %t3 = getelementptr i32, ptr %t2, i32 %t1

  ret ptr %t3

}

attributes #0 = { nounwind optnone noinline }
