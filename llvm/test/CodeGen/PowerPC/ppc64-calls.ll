; RUN: llc < %s -march=ppc64 | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @foo() nounwind readnone noinline {
  ret void
}

define weak void @foo_weak() nounwind {
  ret void
}

; Calls to local function does not require the TOC restore 'nop'
define void @test_direct() nounwind readnone {
; CHECK: test_direct:
  tail call void @foo() nounwind
; CHECK: bl foo
; CHECK-NOT: nop
  ret void
}

; Calls to weak function requires a TOC restore 'nop' because they
; may be overridden in a different module.
define void @test_weak() nounwind readnone {
; CHECK: test_weak:
  tail call void @foo_weak() nounwind
; CHECK: bl foo
; CHECK-NEXT: nop
  ret void
}

; Indirect calls requires a full stub creation
define void @test_indirect(void ()* nocapture %fp) nounwind {
; CHECK: test_indirect:
  tail call void %fp() nounwind
; CHECK: ld [[FP:[0-9]+]], 0(3)
; CHECK: ld 11, 16(3)
; CHECK: ld 2, 8(3)
; CHECK-NEXT: mtctr [[FP]]
; CHECK-NEXT: bctrl
; CHECK-NEXT: ld 2, 40(1)
  ret void
}

; Absolute vales should be have the TOC restore 'nop'
define void @test_abs() nounwind {
; CHECK: test_abs:
  tail call void inttoptr (i64 1024 to void ()*)() nounwind
; CHECK: bla 1024
; CHECK-NEXT: nop
  ret void
}

declare double @sin(double) nounwind

; External functions call should also have a 'nop'
define double @test_external(double %x) nounwind {
; CHECK: test_external:
  %call = tail call double @sin(double %x) nounwind
; CHECK: bl sin
; CHECK-NEXT: nop
  ret double %call
}
