; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -force-vector-width=4 -debug-only=loop-vectorize -disable-output %s 2>&1 | FileCheck %s

define void @narrowed_cast(ptr noalias %src, ptr noalias %dst, i64 %n) {
; CHECK-LABEL: Checking a loop in 'narrowed_cast'
; CHECK: EMIT-SCALAR ir<%conv> = fptosi ir<%uniform_load> to i32
; CHECK: Cost of 1 for VF 4: EMIT-SCALAR ir<%conv> = fptosi ir<%uniform_load> to i32
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %uniform_load = load float, ptr %src, align 4
  %conv = fptosi float %uniform_load to i32
  %gep = getelementptr i32, ptr %dst, i64 %iv
  store i32 %conv, ptr %gep, align 4
  %iv.next = add i64 %iv, 1
  %cmp = icmp ult i64 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

