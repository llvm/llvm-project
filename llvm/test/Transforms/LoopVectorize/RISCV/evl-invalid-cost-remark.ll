; RUN: opt -passes=loop-vectorize -disable-output \
; RUN:   -mtriple=riscv64 -mattr=+v \
; RUN:   -pass-remarks-analysis=loop-vectorize %s 2>&1 | FileCheck %s

; Check that we get a proper remark naming load/store (not a crash,
; and not the wrong recipe name "phi").
; CHECK: Recipe with invalid costs prevented vectorization
; CHECK-SAME: load
; CHECK: Recipe with invalid costs prevented vectorization
; CHECK-SAME: store
; CHECK-NOT: phi

define void @f(ptr noalias %a, ptr noalias %b, i64 %n) minsize {
entry:
  br label %loop
loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %p = getelementptr i32, ptr %b, i64 %iv
  %x = load i32, ptr %p
  %q = getelementptr i32, ptr %a, i64 %iv
  store i32 %x, ptr %q
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop
exit:
  ret void
}
