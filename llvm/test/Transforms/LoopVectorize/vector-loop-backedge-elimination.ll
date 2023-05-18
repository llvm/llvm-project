; RUN: opt -passes=loop-vectorize -force-vector-width=8 -force-vector-interleave=1 -S %s | FileCheck --check-prefixes=CHECK,VF8UF1 %s
; RUN: opt -passes=loop-vectorize -force-vector-width=8 -force-vector-interleave=2 -S %s | FileCheck --check-prefixes=CHECK,VF8UF2 %s
; RUN: opt -passes=loop-vectorize -force-vector-width=16 -force-vector-interleave=1 -S %s | FileCheck --check-prefixes=CHECK,VF16UF1 %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; Check if the vector loop condition can be simplified to true for a given
; VF/IC combination.
define void @test_tc_less_than_16(ptr %A, i64 %N) {
; CHECK-LABEL: define void @test_tc_less_than_16(
; VF8UF1:       [[CMP:%.+]] = icmp eq i64 %index.next, %n.vec
; VF8UF1-NEXT:  br i1 [[CMP]], label %middle.block, label %vector.body
;
; VF8UF2:       br i1 true, label %middle.block, label %vector.body
;
; VF16UF1:      br i1 true, label %middle.block, label %vector.body
;
entry:
  %and = and i64 %N, 15
  br label %loop

loop:
  %iv = phi i64 [ %and, %entry ], [ %iv.next, %loop ]
  %p.src = phi ptr [ %A, %entry ], [ %p.src.next, %loop ]
  %p.src.next = getelementptr inbounds i8, ptr %p.src, i64 1
  %l = load i8, ptr %p.src, align 1
  %add = add nsw i8 %l, 10
  store i8 %add, ptr %p.src
  %iv.next = add nsw i64 %iv, -1
  %cmp = icmp eq i64 %iv.next, 0
  br i1 %cmp, label %exit, label %loop

exit:
  ret void
}
