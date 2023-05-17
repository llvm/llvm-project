; RUN: opt -disable-output -passes=loop-vectorize -pass-remarks-analysis='.*' -force-vector-width=2 2>&1 %s | FileCheck %s

; Make sure LV does not crash when generating remarks for loops with non-unique
; exit blocks.
define i32 @test_non_unique_exit_blocks(ptr nocapture readonly align 4 dereferenceable(1024) %data, i32 %x) {
; CHECK: loop not vectorized: could not determine number of loop iterations
;
entry:
  br label %for.header

for.header:                                         ; preds = %for.cond.lr.ph, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.latch ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 256
  br i1 %exitcond.not, label %header.exit, label %for.latch

for.latch:
  %arrayidx = getelementptr inbounds i32, ptr %data, i64 %iv
  %lv = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp eq i32 %lv, %x
  br i1 %cmp1, label %latch.exit, label %for.header

header.exit:                       ; preds = %for.body
  ret i32 0

latch.exit:                       ; preds = %for.body
  ret i32 %lv
}
