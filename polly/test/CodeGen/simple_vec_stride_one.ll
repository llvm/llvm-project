; RUN: opt %loadPolly -polly-codegen -polly-vectorizer=polly \
; RUN:                 < %s -S | FileCheck %s

; CHECK: store <4 x double> %val.s2a_p_splat, <4 x double>* %vector_ptr
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @update_access_functions(i64 %arg, double* %A, double* %B) {
bb3:
  br label %loop1

loop1:
  %indvar = phi i64 [ %indvar.next, %loop1 ], [ 0, %bb3 ]
  %ptr1 = getelementptr inbounds double, double* %A, i64 %indvar
  store double 42.0, double* %ptr1, align 8
  %indvar.next = add nuw nsw i64 %indvar, 1
  %cmp = icmp ne i64 %indvar.next, 4
  br i1 %cmp, label %loop1, label %loop2

loop2:
  %indvar.2 = phi i64 [ %indvar.2.next, %loop2 ], [ 0, %loop1 ]
  %ptr2 = getelementptr inbounds double, double* %A, i64 %indvar.2
  %val = load double, double* %ptr2, align 8
  %indvar.2.next = add nuw nsw i64 %indvar.2, 1
  %cmp.2 = icmp ne i64 %indvar.2.next, 4
  br i1 %cmp.2, label %loop2, label %loop3

loop3:
  %indvar.3 = phi i64 [ %indvar.3.next, %loop3 ], [ 0, %loop2 ]
  %ptr3 = getelementptr inbounds double, double* %A, i64 %indvar.3
  store double %val, double* %ptr3, align 8
  %indvar.3.next = add nuw nsw i64 %indvar.3, 1
  %cmp.3 = icmp ne i64 %indvar.3.next, 4
  br i1 %cmp.3, label %loop3, label %exit

exit:
  ret void
}
