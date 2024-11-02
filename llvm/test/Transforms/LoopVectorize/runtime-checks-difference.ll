; RUN: opt %s -passes=loop-vectorize -hoist-runtime-checks=false -force-vector-width=4 -force-vector-interleave=1 -S | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

define void @same_step_and_size(ptr %a, ptr %b, i64 %n) {
; CHECK-LABEL: @same_step_and_size(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A2:%.*]] = ptrtoint ptr [[A:%.*]] to i64
; CHECK-NEXT:    [[B1:%.*]] = ptrtoint ptr [[B:%.*]] to i64
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %scalar.ph, label %vector.memcheck
; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[TMP0:%.*]] = sub i64 [[B1]], [[A2]]
; CHECK-NEXT:    [[DIFF_CHECK:%.*]] = icmp ult i64 [[TMP0]], 16
; CHECK-NEXT:    br i1 [[DIFF_CHECK]], label %scalar.ph, label %vector.ph
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  %l = load i32, ptr %gep.a
  %mul = mul nsw i32 %l, 3
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %mul, ptr %gep.b
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @same_step_and_size_no_dominance_between_accesses(ptr %a, ptr %b, i64 %n, i64 %x) {
; CHECK-LABEL: @same_step_and_size_no_dominance_between_accesses(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[B2:%.*]] = ptrtoint ptr [[B:%.*]] to i64
; CHECK-NEXT:    [[A1:%.*]] = ptrtoint ptr [[A:%.*]] to i64
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %scalar.ph, label %vector.memcheck
; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[TMP0:%.*]] = sub i64 [[A1]], [[B2]]
; CHECK-NEXT:    [[DIFF_CHECK:%.*]] = icmp ult i64 [[TMP0]], 16
; CHECK-NEXT:    br i1 [[DIFF_CHECK]], label %scalar.ph, label %vector.ph
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %cmp = icmp ne i64 %iv, %x
  br i1 %cmp, label %then, label %else

then:
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 0, ptr %gep.a
  br label %loop.latch

else:
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 10, ptr %gep.b
  br label %loop.latch

loop.latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @different_steps_and_different_access_sizes(ptr %a, ptr %b, i64 %n) {
; CHECK-LABEL: @different_steps_and_different_access_sizes(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %scalar.ph, label %vector.memcheck
; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[N_SHL_2:%.]] = shl i64 %n, 2
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr i8, ptr %b, i64 [[N_SHL_2]]
; CHECK-NEXT:    [[N_SHL_1:%.]] = shl i64 %n, 1
; CHECK-NEXT:    [[SCEVGEP4:%.*]] = getelementptr i8, ptr %a, i64 [[N_SHL_1]]
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ult ptr %b, [[SCEVGEP4]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ult ptr %a, [[SCEVGEP]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT]], label %scalar.ph, label %vector.ph
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr inbounds i16, ptr %a, i64 %iv
  %l = load i16, ptr %gep.a
  %l.ext = sext i16 %l to i32
  %mul = mul nsw i32 %l.ext, 3
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %mul, ptr %gep.b
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

define void @steps_match_but_different_access_sizes_1(ptr %a, ptr %b, i64 %n) {
; CHECK-LABEL: @steps_match_but_different_access_sizes_1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A2:%.*]] = ptrtoint ptr [[A:%.*]] to i64
; CHECK-NEXT:    [[B1:%.*]] = ptrtoint ptr [[B:%.*]] to i64
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %scalar.ph, label %vector.memcheck
; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[B1]], -2
; CHECK-NEXT:    [[TMP1:%.*]] = sub i64 [[TMP0]], [[A2]]
; CHECK-NEXT:    [[DIFF_CHECK:%.*]] = icmp ult i64 [[TMP1]], 16
; CHECK-NEXT:    br i1 [[DIFF_CHECK]], label %scalar.ph, label %vector.ph
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.a = getelementptr inbounds [2 x i16], ptr %a, i64 %iv, i64 1
  %l = load i16, ptr %gep.a
  %l.ext = sext i16 %l to i32
  %mul = mul nsw i32 %l.ext, 3
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  store i32 %mul, ptr %gep.b
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; Same as @steps_match_but_different_access_sizes_1, but with source and sink
; accesses flipped.
define void @steps_match_but_different_access_sizes_2(ptr %a, ptr %b, i64 %n) {
; CHECK-LABEL: @steps_match_but_different_access_sizes_2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[B2:%.*]] = ptrtoint ptr [[B:%.*]] to i64
; CHECK-NEXT:    [[A1:%.*]] = ptrtoint ptr [[A:%.*]] to i64
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %scalar.ph, label %vector.memcheck
; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[A1]], 2
; CHECK-NEXT:    [[TMP1:%.*]] = sub i64 [[TMP0]], [[B2]]
; CHECK-NEXT:    [[DIFF_CHECK:%.*]] = icmp ult i64 [[TMP1]], 16
; CHECK-NEXT:    br i1 [[DIFF_CHECK]], label %scalar.ph, label %vector.ph
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
  %l = load i32, ptr %gep.b
  %mul = mul nsw i32 %l, 3
  %gep.a = getelementptr inbounds [2 x i16], ptr %a, i64 %iv, i64 1
  %trunc = trunc i32 %mul to i16
  store i16 %trunc, ptr %gep.a
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, %n
  br i1 %exitcond, label %exit, label %loop

exit:
  ret void
}

; Full no-overlap checks are required instead of difference checks, as
; one of the add-recs used is invariant in the inner loop.
; Test case for PR57315.
define void @nested_loop_outer_iv_addrec_invariant_in_inner1(ptr %a, ptr %b, i64 %n) {
; CHECK-LABEL: @nested_loop_outer_iv_addrec_invariant_in_inner1(
; CHECK:        entry:
; CHECK-NEXT:    [[N_SHL_2:%.]] = shl i64 %n, 2
; CHECK-NEXT:    [[B_GEP_UPPER:%.*]] = getelementptr i8, ptr %b, i64 [[N_SHL_2]]
; CHECK-NEXT:    br label %outer

; CHECK:       outer.header:
; CHECK:         [[OUTER_IV_SHL_2:%.]] = shl i64 %outer.iv, 2
; CHECK-NEXT:    [[A_GEP_UPPER:%.*]] = getelementptr i8, ptr %a, i64 [[OUTER_IV_SHL_2]]
; CHECK-NEXT:    [[OUTER_IV_4:%.]] = add i64 [[OUTER_IV_SHL_2]], 4
; CHECK-NEXT:    [[A_GEP_UPPER_4:%.*]] = getelementptr i8, ptr %a, i64 [[OUTER_IV_4]]
; CHECK:         [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %scalar.ph, label %vector.memcheck

; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ult ptr [[A_GEP_UPPER]], [[B_GEP_UPPER]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ult ptr %b, [[A_GEP_UPPER_4]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT]], label %scalar.ph, label %vector.ph
;
entry:
  br label %outer.header

outer.header:
  %outer.iv = phi i64 [ %outer.iv.next, %outer.latch ], [ 0, %entry ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %outer.iv
  br label %inner.body

inner.body:
  %inner.iv = phi i64 [ 0, %outer.header ], [ %inner.iv.next, %inner.body ]
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %inner.iv
  %l = load i32, ptr %gep.b, align 4
  %sub = sub i32 %l, 10
  store i32 %sub, ptr %gep.a, align 4
  %inner.iv.next = add nuw nsw i64 %inner.iv, 1
  %inner.cond = icmp eq i64 %inner.iv.next, %n
  br i1 %inner.cond, label %outer.latch, label %inner.body

outer.latch:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.cond = icmp eq i64 %outer.iv.next, %n
  br i1 %outer.cond, label %exit, label %outer.header

exit:
  ret void
}

; Same as @nested_loop_outer_iv_addrec_invariant_in_inner1 but with dependence
; sink and source swapped.
define void @nested_loop_outer_iv_addrec_invariant_in_inner2(ptr %a, ptr %b, i64 %n) {
; CHECK-LABEL: @nested_loop_outer_iv_addrec_invariant_in_inner2(
; CHECK:        entry:
; CHECK-NEXT:    [[N_SHL_2:%.]] = shl i64 %n, 2
; CHECK-NEXT:    [[B_GEP_UPPER:%.*]] = getelementptr i8, ptr %b, i64 [[N_SHL_2]]
; CHECK-NEXT:    br label %outer

; CHECK:       outer.header:
; CHECK:         [[OUTER_IV_SHL_2:%.]] = shl i64 %outer.iv, 2
; CHECK-NEXT:    [[A_GEP_UPPER:%.*]] = getelementptr i8, ptr %a, i64 [[OUTER_IV_SHL_2]]
; CHECK-NEXT:    [[OUTER_IV_4:%.]] = add i64 [[OUTER_IV_SHL_2]], 4
; CHECK-NEXT:    [[A_GEP_UPPER_4:%.*]] = getelementptr i8, ptr %a, i64 [[OUTER_IV_4]]
; CHECK:         [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N:%.*]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label %scalar.ph, label %vector.memcheck

; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ult ptr %b, [[A_GEP_UPPER_4]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ult ptr [[A_GEP_UPPER]], [[B_GEP_UPPER]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    br i1 [[FOUND_CONFLICT]], label %scalar.ph, label %vector.ph
;
entry:
  br label %outer.header

outer.header:
  %outer.iv = phi i64 [ %outer.iv.next, %outer.latch ], [ 0, %entry ]
  %gep.a = getelementptr inbounds i32, ptr %a, i64 %outer.iv
  br label %inner.body

inner.body:
  %inner.iv = phi i64 [ 0, %outer.header ], [ %inner.iv.next, %inner.body ]
  %l = load i32, ptr %gep.a, align 4
  %sub = sub i32 %l, 10
  %gep.b = getelementptr inbounds i32, ptr %b, i64 %inner.iv
  store i32 %sub, ptr %gep.b, align 4
  %inner.iv.next = add nuw nsw i64 %inner.iv, 1
  %inner.cond = icmp eq i64 %inner.iv.next, %n
  br i1 %inner.cond, label %outer.latch, label %inner.body

outer.latch:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.cond = icmp eq i64 %outer.iv.next, %n
  br i1 %outer.cond, label %exit, label %outer.header

exit:
  ret void
}

; Test case where the AddRec for the pointers in the inner loop have the AddRec
; of the outer loop as start value. It is sufficient to subtract the start
; values (%dst, %src) of the outer AddRecs.
define void @nested_loop_start_of_inner_ptr_addrec_is_same_outer_addrec(ptr nocapture noundef %dst, ptr nocapture noundef readonly %src, i64 noundef %m, i64 noundef %n) {
; CHECK-LABEL: @nested_loop_start_of_inner_ptr_addrec_is_same_outer_addrec(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SRC2:%.*]] = ptrtoint ptr [[SRC:%.*]] to i64
; CHECK-NEXT:    [[DST1:%.*]] = ptrtoint ptr [[DST:%.*]] to i64
; CHECK-NEXT:    [[SUB:%.*]] = sub i64 [[DST1]], [[SRC2]]
; CHECK-NEXT:    br label [[OUTER_LOOP:%.*]]
; CHECK:       outer.loop:
; CHECK-NEXT:    [[OUTER_IV:%.*]] = phi i64 [ 0, [[ENTRY:%.*]] ], [ [[OUTER_IV_NEXT:%.*]], [[INNER_EXIT:%.*]] ]
; CHECK-NEXT:    [[MUL:%.*]] = mul nsw i64 [[OUTER_IV]], [[N]]
; CHECK-NEXT:    [[MIN_ITERS_CHECK:%.*]] = icmp ult i64 [[N]], 4
; CHECK-NEXT:    br i1 [[MIN_ITERS_CHECK]], label [[SCALAR_PH:%.*]], label [[VECTOR_MEMCHECK:%.*]]
; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[DIFF_CHECK:%.*]] = icmp ult i64 [[SUB]], 16
; CHECK-NEXT:    br i1 [[DIFF_CHECK]], label [[SCALAR_PH]], label [[VECTOR_PH:%.*]]
;
entry:
  br label %outer.loop

outer.loop:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %inner.exit ]
  %mul = mul nsw i64 %outer.iv, %n
  br label %inner.loop

inner.loop:
  %iv.inner = phi i64 [ 0, %outer.loop ], [ %iv.inner.next, %inner.loop ]
  %idx = add nuw nsw i64 %iv.inner, %mul
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %idx
  %l = load i32, ptr %gep.src, align 4
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 %idx
  %add = add nsw i32 %l, 10
  store i32 %add, ptr %gep.dst, align 4
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %inner.exit.cond = icmp eq i64 %iv.inner.next, %n
  br i1 %inner.exit.cond, label %inner.exit, label %inner.loop

inner.exit:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.exit.cond = icmp eq i64 %outer.iv.next, %m
  br i1 %outer.exit.cond, label %outer.exit, label %outer.loop

outer.exit:
  ret void
}

define void @use_diff_checks_when_retrying_with_rt_checks(i64 %off, ptr %dst, ptr %src) {
; CHECK-LABEL: @use_diff_checks_when_retrying_with_rt_checks(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label %scalar.ph, label %vector.memcheck
; CHECK:       vector.memcheck:
; CHECK-NEXT:    [[TMP0:%.*]] = shl i64 %off, 3
; CHECK-NEXT:    [[SCEVGEP:%.*]] = getelementptr i8, ptr %dst, i64 [[TMP0]]
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[TMP0]], 8000
; CHECK-NEXT:    [[SCEVGEP1:%.*]] = getelementptr i8, ptr %dst, i64 [[TMP1]]
; CHECK-NEXT:    [[SCEVGEP2:%.*]] = getelementptr i8, ptr %dst, i64 8000
; CHECK-NEXT:    [[SCEVGEP3:%.*]] = getelementptr i8, ptr %src, i64 8000
; CHECK-NEXT:    [[SCEVGEP4:%.*]] = getelementptr i8, ptr %src, i64 8
; CHECK-NEXT:    [[SCEVGEP5:%.*]] = getelementptr i8, ptr %src, i64 8008
; CHECK-NEXT:    [[BOUND0:%.*]] = icmp ult ptr [[SCEVGEP]], [[SCEVGEP2]]
; CHECK-NEXT:    [[BOUND1:%.*]] = icmp ult ptr %dst, [[SCEVGEP1]]
; CHECK-NEXT:    [[FOUND_CONFLICT:%.*]] = and i1 [[BOUND0]], [[BOUND1]]
; CHECK-NEXT:    [[BOUND06:%.*]] = icmp ult ptr [[SCEVGEP]], [[SCEVGEP3]]
; CHECK-NEXT:    [[BOUND17:%.*]] = icmp ult ptr %src, [[SCEVGEP1]]
; CHECK-NEXT:    [[FOUND_CONFLICT8:%.*]] = and i1 [[BOUND06]], [[BOUND17]]
; CHECK-NEXT:    [[CONFLICT_RDX:%.*]] = or i1 [[FOUND_CONFLICT]], [[FOUND_CONFLICT8]]
; CHECK-NEXT:    [[BOUND09:%.*]] = icmp ult ptr [[SCEVGEP]], [[SCEVGEP5]]
; CHECK-NEXT:    [[BOUND110:%.*]] = icmp ult ptr [[SCEVGEP4]], [[SCEVGEP1]]
; CHECK-NEXT:    [[FOUND_CONFLICT11:%.*]] = and i1 [[BOUND09]], [[BOUND110]]
; CHECK-NEXT:    [[CONFLICT_RDX12:%.*]] = or i1 [[CONFLICT_RDX]], [[FOUND_CONFLICT11]]
; CHECK-NEXT:    [[BOUND013:%.*]] = icmp ult ptr %dst, [[SCEVGEP3]]
; CHECK-NEXT:    [[BOUND114:%.*]] = icmp ult ptr %src, [[SCEVGEP2]]
; CHECK-NEXT:    [[FOUND_CONFLICT15:%.*]] = and i1 [[BOUND013]], [[BOUND114]]
; CHECK-NEXT:    [[CONFLICT_RDX16:%.*]] = or i1 [[CONFLICT_RDX12]], [[FOUND_CONFLICT15]]
; CHECK-NEXT:    [[BOUND017:%.*]] = icmp ult ptr %dst, [[SCEVGEP5]]
; CHECK-NEXT:    [[BOUND118:%.*]] = icmp ult ptr [[SCEVGEP4]], [[SCEVGEP2]]
; CHECK-NEXT:    [[FOUND_CONFLICT19:%.*]] = and i1 [[BOUND017]], [[BOUND118]]
; CHECK-NEXT:    [[CONFLICT_RDX20:%.*]] = or i1 [[CONFLICT_RDX16]], [[FOUND_CONFLICT19]]
; CHECK-NEXT:    br i1 [[CONFLICT_RDX20]], label %scalar.ph, label %vector.ph
; CHECK:       vector.ph:
; CHECK-NEXT:    br label %vector.body
;
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.off = add i64 %off, %iv
  %gep.src = getelementptr i64, ptr %src, i64 %iv
  %l.0 = load i64, ptr %gep.src, align 8
  %gep.dst.off = getelementptr i64, ptr %dst, i64 %iv.off
  store i64 %l.0, ptr %gep.dst.off, align 8
  %gep.src.8 = getelementptr i8, ptr %gep.src, i64 8
  %l.1 = load i64, ptr %gep.src.8, align 8
  %gep.dst.iv = getelementptr i64, ptr %dst, i64 %iv
  store i64 %l.1, ptr %gep.dst.iv, align 8
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 1000
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}
