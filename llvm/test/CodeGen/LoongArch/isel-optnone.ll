; REQUIRES: asserts
; RUN: llc %s -O0 -mtriple=loongarch64 -o /dev/null -debug-only=isel 2>&1 | FileCheck %s

define void @fooOptnone() #0 {
; CHECK-NOT: Changing optimization level for Function fooOptnone
; CHECK-NOT: Restoring optimization level for Function fooOptnone
  ret void
}

attributes #0 = { nounwind optnone noinline }
