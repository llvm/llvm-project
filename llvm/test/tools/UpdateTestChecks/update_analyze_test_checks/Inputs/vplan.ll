; RUN: opt -passes=loop-vectorize -vplan-print-after="optimize$" \
; RUN:   -force-vector-width=4 -disable-output %s 2>&1 | FileCheck %s

define void @simple(ptr %p, i64 %n) {
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep = getelementptr i32, ptr %p, i64 %iv
  store i32 0, ptr %gep
  %iv.next = add i64 %iv, 1
  %cmp = icmp ult i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}
