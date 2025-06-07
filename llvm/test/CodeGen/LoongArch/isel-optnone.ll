; REQUIRES: asserts
; RUN: llc %s -O0 -mtriple=loongarch64 -o /dev/null -debug-only=isel 2>&1 | FileCheck %s

define void @fooOptnone() #0 {
; CHECK: Changing optimization level for Function fooOptnone
; CHECK: Before: -O2 ; After: -O0

; CHECK: Restoring optimization level for Function fooOptnone
; CHECK: Before: -O0 ; After: -O2
  ret void
}

attributes #0 = { nounwind optnone noinline }
