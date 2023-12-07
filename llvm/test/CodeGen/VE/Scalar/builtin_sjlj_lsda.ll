; RUN: llc < %s -mtriple=ve -exception-model sjlj | FileCheck %s

; Function Attrs: noinline nounwind optnone
define ptr @test_lsda() {
; CHECK-LABEL: test_lsda:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s0, GCC_except_table0@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, GCC_except_table0@hi(, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %ret = call ptr @llvm.eh.sjlj.lsda()
  ret ptr %ret
}

; Function Attrs: nounwind
declare ptr @llvm.eh.sjlj.lsda()
