; RUN: llc < %s -mtriple=nvptx-unknown-unknown | FileCheck %s
;
; NVPTXTargetLowering::getFunctionParamOptimizedAlign, which was introduces in
; D120129, contained a poorly designed assertion checking that a function with
; internal or private linkage is not a kernel. It relied on invariants that
; were not actually guaranteed, and that resulted in compiler crash with some
; CUDA versions (see discussion with @jdoerfert in D120129). This test contains
; metadata that caused compiler crash and a function with internal linkage
; which purpose is to let compiler run on path where the crash happened.
; Metadata was obtained from libdevice.10.bc shipped with cuda-11-0.


define internal i32 @foo() {
  ; CHECK-LABEL: .func (.param .b32 func_retval0) foo()
  ret i32 42
}

define i32 @bar() {
  ; CHECK-LABEL: .visible .func (.param .b32 func_retval0) bar()
  %x = call i32 @foo()
  ret i32 %x
}

!nvvmir.version = !{!0}
!nvvm.annotations = !{!1, !2, !1, !3, !3, !3, !3, !4, !4, !3}

!0 = !{i32 1, i32 4}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.unroll.count", i32 1}
!7 = distinct !{!7, !6}
