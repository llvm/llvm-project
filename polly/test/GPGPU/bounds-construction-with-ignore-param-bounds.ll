; RUN: opt %loadPolly -S -polly-codegen-ppcg \
; RUN: -polly-ignore-parameter-bounds \
; RUN: -polly-invariant-load-hoisting < %s| FileCheck %s -check-prefix=HOST-IR
;
; REQUIRES: pollyacc

; When we have `-polly-ignore-parameter-bounds`, `Scop::Context` does not contain
; all the parameters present in the program.
;
; The construction of the `isl_multi_pw_aff` requires all the indivisual `pw_aff`
; to have the same parameter dimensions. To achieve this, we used to realign
; every `pw_aff` with `Scop::Context`. However, in conjunction with
; `-polly-ignore-parameter-bounds`, this is now incorrect, since `Scop::Context`
; does not contain all parameters.
;
; We check that Polly does the right thing in this case and sets up the parameter
; dimensions correctly.


; Check that kernel launch is generated in host IR.
; the declare would not be generated unless a call to a kernel exists.
; HOST-IR: declare void @polly_launchKernel(i8*, i32, i32, i32, i32, i32, i8*)
; ModuleID = 'test/GPGPU/bounds-construction-with-ignore-param-bounds.ll'

; C pseudocode
; ------------
; void f(int *arr, long niters, long stride) {
;     for(int i = 0; i < niters; i++) {
;       arr[i * stride] = 1;
;     }
; }

target datalayout = "e-p:64:64:64-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f16:16:16-f32:32:32-f64:64:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @f(i32 *%arr, i64 %niters, i64 %stride) unnamed_addr #1 {
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %loop ]
  %idx = mul nuw nsw i64 %indvar, %stride
  %slot = getelementptr i32, i32* %arr, i64 %idx
  store i32 1, i32* %slot, align 4
  %indvar.next = add nuw nsw i64 %indvar, 1
  %check = icmp sgt i64 %indvar.next, %niters
  br i1 %check, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind uwtable }
