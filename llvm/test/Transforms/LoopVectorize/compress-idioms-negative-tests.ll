; RUN: opt < %s -lv-monotonic-patterns=true -enable-early-exit-vectorization-with-side-effects -force-target-supports-masked-memory-ops -force-vector-width=4 -passes=loop-vectorize -disable-output -pass-remarks-missed=".*" 2>&1 | FileCheck %s

; CHECK: loop not vectorized

; Negative test: Conditional pointer (rather than index) increments are not supported yet (needs LAA support).
define void @test_compress_store_with_pointer(ptr writeonly noalias %init.dst, ptr readonly %src, i32 %c, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %dst = phi ptr [ %init.dst, %entry ], [ %dst.1, %for.inc ]
  %src.ptr = getelementptr inbounds i32, ptr %src, i64 %iv
  %load.src = load i32, ptr %src.ptr, align 4
  %cmp = icmp slt i32 %load.src, %c
  br i1 %cmp, label %if.then, label %for.inc

if.then:
  %dst.inc = getelementptr inbounds i8, ptr %dst, i64 4
  store i32 %load.src, ptr %dst, align 4
  br label %for.inc

for.inc:
  %dst.1 = phi ptr [ %dst.inc, %if.then ], [ %dst, %for.body ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

; CHECK: loop not vectorized

; Negative test: Storing the conditionally incremented phi is invalid.

define void @test_store_conditionally_incremented_value(ptr writeonly noalias %dst, ptr writeonly noalias %dst2, ptr readonly %src, i32 %c, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %idx = phi i32 [ 0, %entry ], [ %idx.1, %for.inc ]
  %src.ptr = getelementptr inbounds i32, ptr %src, i64 %iv
  %load.src = load i32, ptr %src.ptr, align 4
  %cmp = icmp slt i32 %load.src, %c
  br i1 %cmp, label %if.then, label %for.inc

if.then:
  %dst.ptr = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i32 %idx, ptr %dst.ptr, align 4
  %idx.next = add nsw i32 %idx, 1
  br label %for.inc

for.inc:
  %idx.1 = phi i32 [ %idx.next, %if.then ], [ %idx, %for.body ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

; CHECK: loop not vectorized

; Pre-increment is currently not matched as we require one use of the step instruction.
define void @test_pre_increment_compress_store(ptr writeonly noalias %dst, ptr readonly %src, i32 %c, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %idx = phi i64 [ 0, %entry ], [ %idx.1, %for.inc ]
  %src.ptr = getelementptr inbounds i32, ptr %src, i64 %iv
  %load.src = load i32, ptr %src.ptr, align 4
  %cmp = icmp slt i32 %load.src, %c
  br i1 %cmp, label %if.then, label %for.inc

if.then:
  %idx.next = add nsw i64 %idx, 1
  %dst.ptr = getelementptr inbounds i32, ptr %dst, i64 %idx.next
  store i32 %load.src, ptr %dst.ptr, align 4
  br label %for.inc

for.inc:
  %idx.1 = phi i64 [ %idx.next, %if.then ], [ %idx, %for.body ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

; CHECK: the cost-model indicates that vectorization is not beneficial

; Negative test: In this case the %idx is incremented when %cond.val != 0,
; but the store occurs when %cond.val > 100. The store mask does not match the
; PHI mask, so the loop is not vectorized.
define void @compress_mismatched_mask(ptr noalias %dst, ptr noalias %src, ptr noalias %cond, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %idx = phi i64 [ 0, %entry ], [ %idx.next, %for.inc ]
  %cond.ptr = getelementptr inbounds nuw [4 x i8], ptr %cond, i64 %iv
  %cond.val = load i32, ptr %cond.ptr, align 4
  %cond.bool = icmp eq i32 %cond.val, 0
  br i1 %cond.bool, label %for.inc, label %if.then

if.then:
  %cmp.cond = icmp sgt i32 %cond.val, 100
  br i1 %cmp.cond, label %if.then1, label %if.end

if.then1:
  %src.ptr = getelementptr inbounds nuw [4 x i8], ptr %src, i64 %iv
  %src.val = load i32, ptr %src.ptr, align 4
  %dst.ptr = getelementptr inbounds [4 x i8], ptr %dst, i64 %idx
  store i32 %src.val, ptr %dst.ptr, align 4
  br label %if.end

if.end:
  %inc = add nsw i64 %idx, 1
  br label %for.inc

for.inc:
  %idx.next = phi i64 [ %inc, %if.end ], [ %idx, %for.body ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}

; CHECK: the cost-model indicates that vectorization is not beneficial

; Negative test: Simple early exit loop with a compressstore. This fails in VPlan handling for early exits.
define i32 @compress_store_with_early_exit(ptr dereferenceable(1024) %dst, ptr noalias dereferenceable(1024) %src, ptr noalias dereferenceable(1024) %cond, ptr noalias dereferenceable(1024) %exit_cond) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %idx = phi i64 [ 0, %entry ], [ %idx.2, %for.inc ]
  %cond.ptr = getelementptr inbounds nuw i32, ptr %cond, i64 %iv
  %cond.val = load i32, ptr %cond.ptr, align 4
  %compress.cond = icmp eq i32 %cond.val, 0
  %exit.ptr = getelementptr inbounds nuw i32, ptr %exit_cond, i64 %iv
  %exit.val = load i32, ptr %exit.ptr, align 4
  br i1 %compress.cond, label %for.inc, label %if.then

if.then:
  %src.ptr = getelementptr inbounds nuw i32, ptr %src, i64 %iv
  %src.val = load i32, ptr %src.ptr, align 4
  %dst.ptr = getelementptr inbounds i32, ptr %dst, i64 %idx
  store i32 %src.val, ptr %dst.ptr, align 4
  %not.exit.cond = icmp eq i32 %exit.val, 0
  %inc = add nsw i64 %idx, 1
  br i1 %not.exit.cond, label %for.inc, label %early.exit

for.inc:
  %idx.2 = phi i64 [ %idx, %for.body ], [ %inc, %if.then ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 128
  br i1 %exitcond.not, label %early.exit, label %for.body

early.exit:
  %ret = phi i32 [ 1, %if.then ], [ 0, %for.inc ]
  ret i32 %ret
}

; CHECK: loop not vectorized

; Negative test: Using the monotonic phi outside the loop is not supported.
define i64 @out_of_loop_use_of_monotonic_phi(ptr writeonly noalias %dst, ptr readonly %src, i32 %c, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %idx = phi i64 [ 0, %entry ], [ %idx.1, %for.inc ]
  %src.ptr = getelementptr inbounds i32, ptr %src, i64 %iv
  %load.src = load i32, ptr %src.ptr, align 4
  %cmp = icmp slt i32 %load.src, %c
  br i1 %cmp, label %if.then, label %for.inc

if.then:
  %dst.ptr = getelementptr inbounds i32, ptr %dst, i64 %idx
  store i32 %load.src, ptr %dst.ptr, align 4
  %idx.next = add nsw i64 %idx, 1
  br label %for.inc

for.inc:
  %idx.1 = phi i64 [ %idx.next, %if.then ], [ %idx, %for.body ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret i64 %idx
}

; CHECK: loop not vectorized

; Negative test: Matching an extended monotonic phi index is not supported yet.
; Note: We should be able to support this case by using the no-wrap flags on %idx.next.
define void @test_compress_store_with_extended_index(ptr writeonly noalias %dst, ptr readonly %src, i32 %c, i64 %n) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.inc ]
  %idx = phi i32 [ 0, %entry ], [ %idx.1, %for.inc ]
  %src.ptr = getelementptr inbounds i32, ptr %src, i64 %iv
  %load.src = load i32, ptr %src.ptr, align 4
  %cmp = icmp slt i32 %load.src, %c
  br i1 %cmp, label %if.then, label %for.inc

if.then:
  %dst.idx = sext i32 %idx to i64
  %dst.ptr = getelementptr inbounds i32, ptr %dst, i64 %dst.idx
  store i32 %load.src, ptr %dst.ptr, align 4
  %idx.next = add nsw i32 %idx, 1
  br label %for.inc

for.inc:
  %idx.1 = phi i32 [ %idx.next, %if.then ], [ %idx, %for.body ]
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret void
}
