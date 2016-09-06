; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -disable-fp-elim | FileCheck -check-prefix CHECK-WITHFP-ARM64 %s

; Make sure a reasonably sane prologue and epilogue are
; generated. This test is not robust in the face of an frame-handling
; evolving, but still has value for unrelated changes, I
; believe.
;
; In particular, it will fail when ldp/stp are used for frame setup,
; when FP-elim is implemented, and when addressing from FP is
; implemented.

@var = global i64 0
@local_addr = global i64* null

declare void @foo()

define void @trivial_func() nounwind {
; CHECK-LABEL: trivial_func: // @trivial_func
; CHECK-NEXT: // BB#0
; CHECK-NEXT: ret

  ret void
}

define void @trivial_fp_func() {
; CHECK-LABEL: trivial_fp_func:
; CHECK: str x30, [sp, #-16]!
; CHECK-NOT: mov x29, sp

; CHECK-WITHFP-ARM64-LABEL: trivial_fp_func:
; CHECK-WITHFP-ARM64: stp x29, x30, [sp, #-16]!
; CHECK-WITHFP-ARM64-NEXT: mov x29, sp

; Dont't really care, but it would be a Bad Thing if this came after the epilogue.
; CHECK-WITHFP-ARM64: bl foo
; CHECK: bl foo
  call void @foo()
  ret void

; CHECK: ldr x30, [sp], #16
; CHECK-NEXT: ret

; CHECK-WITHFP-ARM64: ldp x29, x30, [sp], #16
; CHECK-WITHFP-ARM64-NEXT: ret
}

define void @stack_local() {
  %local_var = alloca i64
; CHECK-LABEL: stack_local:
; CHECK: sub sp, sp, #16

  %val = load i64, i64* @var
  store i64 %val, i64* %local_var
; CHECK-DAG: str {{x[0-9]+}}, [sp, #{{[0-9]+}}]

  store i64* %local_var, i64** @local_addr
; CHECK-DAG: add {{x[0-9]+}}, sp, #{{[0-9]+}}

  ret void
}
