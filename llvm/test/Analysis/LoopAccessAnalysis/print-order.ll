; REQUIRES: asserts
; RUN: opt -passes='print<access-info>' -debug-only=loop-accesses -disable-output < %s 2>&1 | FileCheck %s

; void negative_step(int *A) {
;  for (int i = 1022; i >= 0; i--)
;    A[i+1] = A[i] + 1;
; }

; CHECK: LAA: Found a loop in negative_step: loop
; CHECK: LAA: Checking memory dependencies
; CHECK-NEXT: LAA: Src Scev: {(4092 + %A),+,-4}<nw><%loop>Sink Scev: {(4088 + %A)<nuw>,+,-4}<nw><%loop>(Induction step: -1)
; CHECK-NEXT: LAA: Distance for   store i32 %add, ptr %gep.A.plus.1, align 4 to   %l = load i32, ptr %gep.A, align 4: -4
; CHECK-NEXT: LAA: Dependence is negative

define void @negative_step(ptr nocapture %A) {
entry:
  %A.plus.1 = getelementptr i32, ptr %A, i64 1
  br label %loop

loop:
  %iv = phi i64 [ 1022, %entry ], [ %iv.next, %loop ]
  %gep.A = getelementptr inbounds i32, ptr %A, i64 %iv
  %l = load i32, ptr %gep.A, align 4
  %add = add nsw i32 %l, 1
  %gep.A.plus.1 = getelementptr i32, ptr %A.plus.1, i64 %iv
  store i32 %add, ptr %gep.A.plus.1, align 4
  %iv.next = add nsw i64 %iv, -1
  %cmp.not = icmp eq i64 %iv, 0
  br i1 %cmp.not, label %exit, label %loop

exit:
  ret void
}

; void positive_step(int *A) {
;  for (int i = 1; i < 1024; i++)
;    A[i-1] = A[i] + 1;
; }

; CHECK: LAA: Found a loop in positive_step: loop
; CHECK: LAA: Checking memory dependencies
; CHECK-NEXT: LAA: Src Scev: {(4 + %A)<nuw>,+,4}<nuw><%loop>Sink Scev: {%A,+,4}<nw><%loop>(Induction step: 1)
; CHECK-NEXT: LAA: Distance for   %l = load i32, ptr %gep.A, align 4 to   store i32 %add, ptr %gep.A.minus.1, align 4: -4
; CHECK-NEXT: LAA: Dependence is negative

define void @positive_step(ptr nocapture %A) {
entry:
  %A.minus.1 = getelementptr i32, ptr %A, i64 -1
  br label %loop

loop:
  %iv = phi i64 [ 1, %entry ], [ %iv.next, %loop ]
  %gep.A = getelementptr inbounds i32, ptr %A, i64 %iv
  %l = load i32, ptr %gep.A, align 4
  %add = add nsw i32 %l, 1
  %gep.A.minus.1 = getelementptr i32, ptr %A.minus.1, i64 %iv
  store i32 %add, ptr %gep.A.minus.1, align 4
  %iv.next = add nsw i64 %iv, 1
  %cmp.not = icmp eq i64 %iv, 1024
  br i1 %cmp.not, label %exit, label %loop

exit:
  ret void
}

