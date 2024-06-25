; RUN: opt %loadNPMPolly '-passes=polly-import-jscop,polly-codegen' \
; RUN:                -polly-import-jscop-postfix=transformed \
; RUN:                 < %s -S | FileCheck %s

; CHECK-LABEL: polly.stmt.loop1:
; CHECK-NEXT:   %3 = mul nsw i64 5, %polly.indvar{{[0-9]*}}
; CHECK-NEXT:   %4 = sub nsw i64 %3, 10
; CHECK-NEXT:   %polly.access.A = getelementptr double, ptr %A, i64 %4
; CHECK-NEXT:   store double 4.200000e+01, ptr %polly.access.A, align 8

; CHECK-LABEL: polly.stmt.loop2:
; CHECK-NEXT:   %polly.access.A[[Num0:[0-9]*]] = getelementptr double, ptr %A, i64 42
; CHECK-NEXT:   %val_p_scalar_ = load double, ptr %polly.access.A[[Num0]], align 8

; CHECK-LABEL: polly.stmt.loop3:
; CHECK-NEXT:   %val.s2a.reload = load double, ptr %val.s2a
; CHECK-NEXT:   [[REG0:%.*]] = mul nsw i64 13, %polly.indvar{{[0-9]*}}
; CHECK-NEXT:   [[REG1:%.*]] = add nsw i64 [[REG0]], 5
; CHECK-NEXT:   %polly.access.A[[Num1:[0-9]*]] = getelementptr double, ptr %A, i64 [[REG1]]
; CHECK-NEXT:   store double %val.s2a.reload, ptr %polly.access.A[[Num1]], align 8,

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @update_access_functions(i64 %arg, ptr %A) {
bb3:
  br label %loop1

loop1:
  %indvar = phi i64 [ %indvar.next, %loop1 ], [ 1, %bb3 ]
  %ptr1 = getelementptr inbounds double, ptr %A, i64 %indvar
  store double 42.0, ptr %ptr1, align 8
  %indvar.next = add nuw nsw i64 %indvar, 1
  %cmp = icmp ne i64 %indvar.next, %arg
  br i1 %cmp, label %loop1, label %loop2

loop2:
  %indvar.2 = phi i64 [ %indvar.2.next, %loop2 ], [ 1, %loop1 ]
  %ptr2 = getelementptr inbounds double, ptr %A, i64 %indvar.2
  %val = load double, ptr %ptr2, align 8
  %indvar.2.next = add nuw nsw i64 %indvar.2, 1
  %cmp.2 = icmp ne i64 %indvar.2.next, %arg
  br i1 %cmp.2, label %loop2, label %loop3

loop3:
  %indvar.3 = phi i64 [ %indvar.3.next, %loop3 ], [ 1, %loop2 ]
  %ptr3 = getelementptr inbounds double, ptr %A, i64 %indvar.3
  store double %val, ptr %ptr3, align 8
  %indvar.3.next = add nuw nsw i64 %indvar.3, 1
  %cmp.3 = icmp ne i64 %indvar.3.next, %arg
  br i1 %cmp.3, label %loop3, label %exit

exit:
  ret void
}
