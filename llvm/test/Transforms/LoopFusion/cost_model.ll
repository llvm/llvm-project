; REQUIRES: asserts
; RUN: opt -passes=loop-simplify,loop-fusion -disable-output -stats < %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: opt -passes=loop-simplify,loop-fusion -loop-fusion-cost-model \
; RUN:   -loop-fusion-min-reused-values=0 -disable-output -stats < %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=MODEL

; DEFAULT: 1 loop-fusion - Loops fused
; MODEL: 1 loop-fusion - Loops fused

; C source:
; for (i = 0; i < n; ++i)
;   A[i] = x;
; for (i = 0; i < n; ++i)
;   A[i] += y;

define void @cost_model(ptr noalias %A, float %x, float %y, i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1.latch ]
  %a.gep1 = getelementptr inbounds float, ptr %A, i64 %i1
  store float %x, ptr %a.gep1, align 4
  br label %loop1.latch

loop1.latch:
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2.latch ]
  %a.gep2 = getelementptr inbounds float, ptr %A, i64 %i2
  %old = load float, ptr %a.gep2, align 4
  %sum = fadd float %old, %y
  store float %sum, ptr %a.gep2, align 4
  br label %loop2.latch

loop2.latch:
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}
