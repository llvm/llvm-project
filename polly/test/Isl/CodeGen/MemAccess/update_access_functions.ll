; RUN: opt %loadPolly -polly-import-jscop -polly-import-jscop-dir=%S \
; RUN:                -polly-import-jscop-postfix=transformed -polly-codegen \
; RUN:                 < %s -S | FileCheck %s

; CHECK: polly.stmt.loop2:
; CHECK-NEXT:   %polly.access.A = getelementptr double, double* %A, i64 42
; CHECK-NEXT:   %val_p_scalar_ = load double, double* %polly.access.A

; CHECK: polly.stmt.loop3:
; CHECK-NEXT:   %val.s2a.reload = load double, double* %val.s2a
; CHECK-NEXT:   %scevgep[[R21:[0-9]*]] = getelementptr double, double* %scevgep{{[0-9]*}}, i64 %polly.indvar16
; CHECK-NEXT:   store double %val.s2a.reload, double* %scevgep[[R21]]

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @update_access_functions(i64 %arg, double* %A) {
bb3:
  br label %loop1

loop1:
  %indvar = phi i64 [ %indvar.next, %loop1 ], [ 1, %bb3 ]
  %ptr1 = getelementptr inbounds double, double* %A, i64 %indvar
  store double 42.0, double* %ptr1, align 8
  %indvar.next = add nuw nsw i64 %indvar, 1
  %cmp = icmp ne i64 %indvar.next, %arg
  br i1 %cmp, label %loop1, label %loop2

loop2:
  %indvar.2 = phi i64 [ %indvar.2.next, %loop2 ], [ 1, %loop1 ]
  %ptr2 = getelementptr inbounds double, double* %A, i64 %indvar.2
  %val = load double, double* %ptr2, align 8
  %indvar.2.next = add nuw nsw i64 %indvar.2, 1
  %cmp.2 = icmp ne i64 %indvar.2.next, %arg
  br i1 %cmp.2, label %loop2, label %loop3

loop3:
  %indvar.3 = phi i64 [ %indvar.3.next, %loop3 ], [ 1, %loop2 ]
  %ptr3 = getelementptr inbounds double, double* %A, i64 %indvar.3
  store double %val, double* %ptr3, align 8
  %indvar.3.next = add nuw nsw i64 %indvar.3, 1
  %cmp.3 = icmp ne i64 %indvar.3.next, %arg
  br i1 %cmp.3, label %loop3, label %exit

exit:
  ret void
}
