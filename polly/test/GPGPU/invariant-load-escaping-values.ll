; RUN: opt %loadPolly -S -polly-codegen-ppcg \
; RUN: -polly-invariant-load-hoisting < %s | FileCheck %s

; REQUIRES: pollyacc

; CHECK: store i64 %polly.access.B.load, i64* %invariant.preload.s2a
; CHECK: %invariant.final_reload = load i64, i64* %invariant.preload.s2a

; Verify that the final reload of an invariant scalar memory access uses the
; same stack slot that into which the invariant memory access was stored
; originally. Earlier, this was broken as we introduce a new stack slot aside
; of the preload stack slot, which remained uninitialized and caused our escaping
; loads to contain garbage.

define i64 @foo(float* %A, i64* %B) {
entry:
  br label %loop

loop:
  %indvar = phi i64 [0, %entry], [%indvar.next, %loop]
  %indvar.next = add nsw i64 %indvar, 1
  %idx = getelementptr float, float* %A, i64 %indvar
  store float 42.0, float* %idx
  %invariant = load i64, i64* %B
  %cmp = icmp sle i64 %indvar, 1024
  br i1 %cmp, label %loop, label %exit

exit:
  ret i64 %invariant
}
