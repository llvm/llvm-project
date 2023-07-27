; RUN: opt -S -disable-output -passes='print<access-info>' < %s 2>&1 | FileCheck %s

; Generated from following C program:
; void foo(int len, int *a) {
;   for (int k = 0; k < len; k+=3) {
;     a[k] = a[k + 4];
;     a[k+2] = a[k+6];
;   }
; }
define void @foo(i64  %len, ptr %a) {
; CHECK-LABEL: Loop access info in function 'foo':
; CHECK-NEXT:  loop:
; CHECK-NEXT:    Memory dependences are safe with a maximum dependence distance of 24 bytes
; CHECK-NEXT:    Dependences:
; CHECK-NEXT:      BackwardVectorizable:
; CHECK-NEXT:          store i32 %0, ptr %arrayidx2, align 4 ->
; CHECK-NEXT:          %1 = load i32, ptr %arrayidx5, align 4
; CHECK-EMPTY:
; CHECK-NEXT:    Run-time memory checks:
; CHECK-NEXT:    Grouped accesses:
; CHECK-EMPTY:
; CHECK-NEXT:    Non vectorizable stores to invariant address were not found in loop.
; CHECK-NEXT:    SCEV assumptions:
; CHECK-EMPTY:
; CHECK-NEXT:    Expressions re-written:
;
loop.preheader:
  br label %loop

loop.exit:
  br label %exit

exit:
  ret void

loop:
  %iv = phi i64 [ 0, %loop.preheader ], [ %iv.next, %loop ]
  %iv.4 = add nuw nsw i64 %iv, 4
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv.4
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %0, ptr %arrayidx2, align 4
  %iv.6 = add nuw nsw i64 %iv, 6
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %iv.6
  %1 = load i32, ptr %arrayidx5, align 4
  %iv.2 = add nuw nsw i64 %iv, 2
  %arrayidx8 = getelementptr inbounds i32, ptr %a, i64 %iv.2
  store i32 %1, ptr %arrayidx8, align 4
  %iv.next = add nuw nsw i64 %iv, 3
  %cmp = icmp ult i64 %iv.next, %len
  br i1 %cmp, label %loop, label %loop.exit
}
