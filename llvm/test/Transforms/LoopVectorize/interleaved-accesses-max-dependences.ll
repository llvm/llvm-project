; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -enable-interleaved-mem-accesses=true -max-dependences=0 -S %s | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; None of these stores have dependences between them, so we can successfully
; interleave them even though the max-dependences threshold is 0.
define void @three_interleaved_stores(ptr %arr) {
; CHECK-LABEL: define void @three_interleaved_stores
; CHECK:   store <12 x i8>
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %i.plus.1 = add nuw nsw i64 %i, 1
  %i.plus.2 = add nuw nsw i64 %i, 2
  %gep.i.plus.0 = getelementptr inbounds i8, ptr %arr, i64 %i
  %gep.i.plus.1 = getelementptr inbounds i8, ptr %arr, i64 %i.plus.1
  %gep.i.plus.2 = getelementptr inbounds i8, ptr %arr, i64 %i.plus.2
  store i8 1, ptr %gep.i.plus.0
  store i8 1, ptr %gep.i.plus.1
  store i8 1, ptr %gep.i.plus.2
  %i.next = add nuw nsw i64 %i, 3
  %icmp = icmp ugt i64 %i, 1032
  br i1 %icmp, label %exit, label %loop

exit:
  ret void
}
