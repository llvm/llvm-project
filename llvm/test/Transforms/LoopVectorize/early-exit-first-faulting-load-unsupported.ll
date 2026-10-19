; RUN: opt < %s -p loop-vectorize -force-vector-width=4 -enable-first-faulting-load \
; RUN:   -enable-early-exit-vectorization -pass-remarks-analysis=loop-vectorize \
; RUN:   -S 2>&1 | FileCheck %s

; Without a target that can lower llvm.vp.load.ff, its cost is invalid, so a
; potentially faulting load in an early-exit loop prevents vectorization even
; when first-faulting loads are enabled.

; CHECK: remark: {{.*}} UserVF ignored because of invalid costs.
; CHECK: remark: {{.*}} Recipe with invalid costs prevented vectorization at VF=({{.*}}): load
; CHECK-LABEL: define i64 @early_exit_first_faulting_load(
; CHECK-NOT:   vp.load.ff
; CHECK-NOT:   vector.body
; CHECK:       ret i64
define i64 @early_exit_first_faulting_load(ptr %cond) {
entry:
  br label %loop.body

loop.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %gep.cond = getelementptr [4 x i8], ptr %cond, i64 %iv
  %cond.val = load i32, ptr %gep.cond, align 4
  %early.cond = icmp slt i32 %cond.val, 0
  br i1 %early.cond, label %exit, label %loop.latch

loop.latch:
  %iv.next = add i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 100
  br i1 %exitcond, label %exit, label %loop.body

exit:
  %iv.lcssa = phi i64 [ %iv, %loop.body ], [ %iv, %loop.latch ]
  ret i64 %iv.lcssa
}
