target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; RUN: opt < %s -basicaa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; ptr_phi and ptr2_phi do not alias.
; CHECK: NoAlias: i32* %ptr2_phi, i32* %ptr_phi

define i32 @test_noalias(i32* %ptr2, i32 %count, i32* %coeff) {
entry:
  %ptr = getelementptr inbounds i32* %ptr2, i64 1
  br label %while.body

while.body:
  %num = phi i32 [ %count, %entry ], [ %dec, %while.body ]
  %ptr_phi = phi i32* [ %ptr, %entry ], [ %ptr_inc, %while.body ]
  %ptr2_phi = phi i32* [ %ptr2, %entry ], [ %ptr2_inc, %while.body ]
  %result.09 = phi i32 [ 0 , %entry ], [ %add, %while.body ]
  %dec = add nsw i32 %num, -1
  %0 = load i32* %ptr_phi, align 4
  store i32 %0, i32* %ptr2_phi, align 4
  %1 = load i32* %coeff, align 4
  %2 = load i32* %ptr_phi, align 4
  %mul = mul nsw i32 %1, %2
  %add = add nsw i32 %mul, %result.09
  %tobool = icmp eq i32 %dec, 0
  %ptr_inc = getelementptr inbounds i32* %ptr_phi, i64 1
  %ptr2_inc = getelementptr inbounds i32* %ptr2_phi, i64 1
  br i1 %tobool, label %the_exit, label %while.body

the_exit:
  ret i32 %add
}
