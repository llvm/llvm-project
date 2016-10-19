; RUN: opt < %s -indvars -indvars-post-increment-ranges -S | FileCheck %s

target datalayout = "p:64:64:64-n32:64"

; When the IV in this loop is widened we want to widen this use as well:
; icmp slt i32 %i.inc, %limit
; In order to do this indvars need to prove that the narrow IV def (%i.inc)
; is not-negative from the range check inside of the loop.
define void @test(i32* %base, i32 %limit, i32 %start) {
; CHECK-LABEL: @test(
; CHECK-NOT: trunc

for.body.lr.ph:
  br label %for.body

for.body:
  %i = phi i32 [ %start, %for.body.lr.ph ], [ %i.inc, %for.inc ]
  %within_limits = icmp ult i32 %i, 64
  br i1 %within_limits, label %continue, label %for.end

continue:
  %i.i64 = zext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 %i.i64
  %val = load i32, i32* %arrayidx, align 4
  br label %for.inc

for.inc:
  %i.inc = add nsw nuw i32 %i, 1
  %cmp = icmp slt i32 %i.inc, %limit
  br i1 %cmp, label %for.body, label %for.end

for.end:
  br label %exit

exit:
  ret void
}

define void @test_false_edge(i32* %base, i32 %limit, i32 %start) {
; CHECK-LABEL: @test_false_edge(
; CHECK-NOT: trunc

for.body.lr.ph:
  br label %for.body

for.body:
  %i = phi i32 [ %start, %for.body.lr.ph ], [ %i.inc, %for.inc ]
  %out_of_bounds = icmp ugt i32 %i, 64
  br i1 %out_of_bounds, label %for.end, label %continue

continue:
  %i.i64 = zext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 %i.i64
  %val = load i32, i32* %arrayidx, align 4
  br label %for.inc

for.inc:
  %i.inc = add nsw nuw i32 %i, 1
  %cmp = icmp slt i32 %i.inc, %limit
  br i1 %cmp, label %for.body, label %for.end

for.end:
  br label %exit

exit:
  ret void
}

define void @test_range_metadata(i32* %array_length_ptr, i32* %base,
                                 i32 %limit, i32 %start) {
; CHECK-LABEL: @test_range_metadata(
; CHECK-NOT: trunc

for.body.lr.ph:
  br label %for.body

for.body:
  %i = phi i32 [ %start, %for.body.lr.ph ], [ %i.inc, %for.inc ]
  %array_length = load i32, i32* %array_length_ptr, !range !{i32 0, i32 64 }
  %within_limits = icmp ult i32 %i, %array_length
  br i1 %within_limits, label %continue, label %for.end

continue:
  %i.i64 = zext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 %i.i64
  %val = load i32, i32* %arrayidx, align 4
  br label %for.inc

for.inc:
  %i.inc = add nsw nuw i32 %i, 1
  %cmp = icmp slt i32 %i.inc, %limit
  br i1 %cmp, label %for.body, label %for.end

for.end:
  br label %exit

exit:
  ret void
}

; Negative version of the test above, we don't know anything about
; array_length_ptr range.
define void @test_neg(i32* %array_length_ptr, i32* %base,
                      i32 %limit, i32 %start) {
; CHECK-LABEL: @test_neg(
; CHECK: trunc i64

for.body.lr.ph:
  br label %for.body

for.body:
  %i = phi i32 [ %start, %for.body.lr.ph ], [ %i.inc, %for.inc ]
  %array_length = load i32, i32* %array_length_ptr
  %within_limits = icmp ult i32 %i, %array_length
  br i1 %within_limits, label %continue, label %for.end

continue:
  %i.i64 = zext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 %i.i64
  %val = load i32, i32* %arrayidx, align 4
  br label %for.inc

for.inc:
  %i.inc = add nsw nuw i32 %i, 1
  %cmp = icmp slt i32 %i.inc, %limit
  br i1 %cmp, label %for.body, label %for.end

for.end:
  br label %exit

exit:
  ret void
}

define void @test_transitive_use(i32* %base, i32 %limit, i32 %start) {
; CHECK-LABEL: @test_transitive_use(
; CHECK-NOT: trunc
; CHECK: %result = icmp slt i64

for.body.lr.ph:
  br label %for.body

for.body:
  %i = phi i32 [ %start, %for.body.lr.ph ], [ %i.inc, %for.inc ]
  %within_limits = icmp ult i32 %i, 64
  br i1 %within_limits, label %continue, label %for.end

continue:
  %i.mul.3 = mul nsw nuw i32 %i, 3
  %mul_within = icmp ult i32 %i.mul.3, 64
  br i1 %mul_within, label %guarded, label %continue.2
  
guarded:
  %i.mul.3.inc = add nsw nuw i32 %i.mul.3, 1
  %result = icmp slt i32 %i.mul.3.inc, %limit
  br i1 %result, label %continue.2, label %for.end

continue.2:
  %i.i64 = zext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 %i.i64
  %val = load i32, i32* %arrayidx, align 4
  br label %for.inc

for.inc:
  %i.inc = add nsw nuw i32 %i, 1
  %cmp = icmp slt i32 %i.inc, %limit
  br i1 %cmp, label %for.body, label %for.end


for.end:
  br label %exit

exit:
  ret void
}

declare void @llvm.experimental.guard(i1, ...)

define void @test_guard_one_bb(i32* %base, i32 %limit, i32 %start) {
; CHECK-LABEL: @test_guard_one_bb(
; CHECK-NOT: trunc
; CHECK-NOT: icmp slt i32

for.body.lr.ph:
  br label %for.body

for.body:
  %i = phi i32 [ %start, %for.body.lr.ph ], [ %i.inc, %for.body ]
  %within_limits = icmp ult i32 %i, 64
  %i.i64 = zext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 %i.i64
  %val = load i32, i32* %arrayidx, align 4
  call void(i1, ...) @llvm.experimental.guard(i1 %within_limits) [ "deopt"() ]
  %i.inc = add nsw nuw i32 %i, 1
  %cmp = icmp slt i32 %i.inc, %limit
  br i1 %cmp, label %for.body, label %for.end

for.end:
  br label %exit

exit:
  ret void
}

define void @test_guard_in_the_same_bb(i32* %base, i32 %limit, i32 %start) {
; CHECK-LABEL: @test_guard_in_the_same_bb(
; CHECK-NOT: trunc
; CHECK-NOT: icmp slt i32

for.body.lr.ph:
  br label %for.body

for.body:
  %i = phi i32 [ %start, %for.body.lr.ph ], [ %i.inc, %for.inc ]
  %within_limits = icmp ult i32 %i, 64
  %i.i64 = zext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 %i.i64
  %val = load i32, i32* %arrayidx, align 4
  br label %for.inc

for.inc:
  call void(i1, ...) @llvm.experimental.guard(i1 %within_limits) [ "deopt"() ]
  %i.inc = add nsw nuw i32 %i, 1
  %cmp = icmp slt i32 %i.inc, %limit
  br i1 %cmp, label %for.body, label %for.end

for.end:
  br label %exit

exit:
  ret void
}

define void @test_guard_in_idom(i32* %base, i32 %limit, i32 %start) {
; CHECK-LABEL: @test_guard_in_idom(
; CHECK-NOT: trunc
; CHECK-NOT: icmp slt i32

for.body.lr.ph:
  br label %for.body

for.body:
  %i = phi i32 [ %start, %for.body.lr.ph ], [ %i.inc, %for.inc ]
  %within_limits = icmp ult i32 %i, 64
  call void(i1, ...) @llvm.experimental.guard(i1 %within_limits) [ "deopt"() ]
  %i.i64 = zext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 %i.i64
  %val = load i32, i32* %arrayidx, align 4
  br label %for.inc

for.inc:
  %i.inc = add nsw nuw i32 %i, 1
  %cmp = icmp slt i32 %i.inc, %limit
  br i1 %cmp, label %for.body, label %for.end

for.end:
  br label %exit

exit:
  ret void
}

define void @test_guard_merge_ranges(i32* %base, i32 %limit, i32 %start) {
; CHECK-LABEL: @test_guard_merge_ranges(
; CHECK-NOT: trunc
; CHECK-NOT: icmp slt i32

for.body.lr.ph:
  br label %for.body

for.body:
  %i = phi i32 [ %start, %for.body.lr.ph ], [ %i.inc, %for.body ]
  %within_limits.1 = icmp ult i32 %i, 64
  call void(i1, ...) @llvm.experimental.guard(i1 %within_limits.1) [ "deopt"() ]
  %within_limits.2 = icmp ult i32 %i, 2147483647
  call void(i1, ...) @llvm.experimental.guard(i1 %within_limits.2) [ "deopt"() ]
  %i.i64 = zext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %base, i64 %i.i64
  %val = load i32, i32* %arrayidx, align 4
  %i.inc = add nsw nuw i32 %i, 1
  %cmp = icmp slt i32 %i.inc, %limit
  br i1 %cmp, label %for.body, label %for.end

for.end:
  br label %exit

exit:
  ret void
}
