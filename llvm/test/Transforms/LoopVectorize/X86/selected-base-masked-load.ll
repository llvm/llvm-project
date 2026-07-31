; RUN: opt -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=tigerlake \
; RUN:   -force-vector-width=8 -force-vector-interleave=1 -S %s | FileCheck %s
; RUN: opt -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=tigerlake \
; RUN:   -force-vector-width=8 -force-vector-interleave=1 \
; RUN:   -tail-folding-policy=must-fold-tail -force-tail-folding-style=data \
; RUN:   -S %s | FileCheck %s --check-prefix=TAIL
; RUN: opt -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=tigerlake \
; RUN:   -force-vector-width=8 -force-vector-interleave=1 -disable-output \
; RUN:   -vplan-print-after=widenSelectedBaseLoads %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=VPLAN
; RUN: opt -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=tigerlake \
; RUN:   -S %s | FileCheck %s --check-prefix=ISSUE
; RUN: opt -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=tigerlake \
; RUN:   -force-vector-width=8 -force-vector-interleave=2 -S %s | \
; RUN:   FileCheck %s --check-prefix=UNROLL
; RUN: opt -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=tigerlake \
; RUN:   -force-vector-width=8 -force-vector-interleave=1 -disable-output \
; RUN:   -vplan-print-after=dropPoisonGeneratingRecipes %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=VPLAN-DROP


; A load from a per-lane selected base and a unit-stride index is cheaper as
; two complementary masked loads than as a gather on AVX-512 targets.

define void @selected_base_unit_stride(ptr noalias readonly %conditions,
                                      ptr noalias writeonly %out,
                                      ptr noalias readonly %a,
                                      ptr noalias readonly %b, i64 %n) {
; CHECK-LABEL: @selected_base_unit_stride(
; CHECK: vector.body:
; CHECK: [[MASK:%.*]] = icmp sgt <8 x i32> {{.*}}, zeroinitializer
; CHECK: [[A_PTR:%.*]] = getelementptr i32, ptr %a, i64 {{.*}}
; CHECK: [[B_PTR:%.*]] = getelementptr i32, ptr %b, i64 {{.*}}
; CHECK: [[NOT_MASK:%.*]] = xor <8 x i1> [[MASK]], splat (i1 true)
; CHECK: [[FROM_A:%.*]] = call <8 x i32> @llvm.masked.load.v8i32.p0(ptr align 1 [[A_PTR]], <8 x i1> [[MASK]], <8 x i32> poison)
; CHECK: [[FROM_B:%.*]] = call <8 x i32> @llvm.masked.load.v8i32.p0(ptr align 1 [[B_PTR]], <8 x i1> [[NOT_MASK]], <8 x i32> poison)
; CHECK: [[RESULT:%.*]] = select <8 x i1> [[MASK]], <8 x i32> [[FROM_A]], <8 x i32> [[FROM_B]]
; CHECK-NOT: @llvm.masked.gather
;
; TAIL-LABEL: @selected_base_unit_stride(
; TAIL: [[ACTIVE:%.*]] = call <8 x i1> @llvm.get.active.lane.mask.v8i1.i64(
; TAIL: [[MASK:%.*]] = icmp sgt <8 x i32> {{.*}}, zeroinitializer
; TAIL: [[NOT_MASK:%.*]] = xor <8 x i1> [[MASK]], splat (i1 true)
; TAIL: [[ACTIVE_A:%.*]] = select <8 x i1> [[ACTIVE]], <8 x i1> [[MASK]], <8 x i1> zeroinitializer
; TAIL: [[ACTIVE_B:%.*]] = select <8 x i1> [[ACTIVE]], <8 x i1> [[NOT_MASK]], <8 x i1> zeroinitializer
; TAIL: [[FROM_A:%.*]] = call <8 x i32> @llvm.masked.load.v8i32.p0(ptr align 1 {{.*}}, <8 x i1> [[ACTIVE_A]], <8 x i32> poison)
; TAIL: [[FROM_B:%.*]] = call <8 x i32> @llvm.masked.load.v8i32.p0(ptr align 1 {{.*}}, <8 x i1> [[ACTIVE_B]], <8 x i32> poison)
; TAIL: select <8 x i1> [[MASK]], <8 x i32> [[FROM_A]], <8 x i32> [[FROM_B]]
;
; VPLAN-LABEL: VPlan for loop in 'selected_base_unit_stride' after widenSelectedBaseLoads
; VPLAN: EMIT ir<%src.ptr>.1 = getelementptr inbounds ir<%a>, ir<%iv>
; VPLAN: EMIT ir<%src.ptr>.2 = getelementptr inbounds ir<%b>, ir<%iv>
; VPLAN: [[A_PTR:vp<%[0-9]+>]] = vector-pointer inbounds i32, ir<%src.ptr>.1, ir<1>
; VPLAN: [[B_PTR:vp<%[0-9]+>]] = vector-pointer inbounds i32, ir<%src.ptr>.2, ir<1>
; VPLAN: [[NOT_MASK:vp<%[0-9]+>]] = not ir<%cmp>
; VPLAN: WIDEN ir<%value> = load [[A_PTR]], ir<%cmp>
; VPLAN: WIDEN ir<%value>.1 = load [[B_PTR]], [[NOT_MASK]]
; VPLAN: EMIT {{.*}} = select ir<%cmp>, ir<%value>, ir<%value>.1
;
; Cloning the recipes for multiple parts must preserve the conservative
; alignment selected by this transform.
; UNROLL-LABEL: @selected_base_unit_stride(
; UNROLL-COUNT-4: call <8 x i32> @llvm.masked.load.v8i32.p0(ptr align 1
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %condition.ptr = getelementptr inbounds i32, ptr %conditions, i64 %iv
  %condition = load i32, ptr %condition.ptr, align 4
  %cmp = icmp sgt i32 %condition, 0
  %base = select i1 %cmp, ptr %a, ptr %b
  %src.ptr = getelementptr inbounds i32, ptr %base, i64 %iv
  ; The selected address is known to be 8-byte aligned, but that does not imply
  ; the unselected candidate base has the same alignment.
  %value = load i32, ptr %src.ptr, align 8
  %out.ptr = getelementptr inbounds i32, ptr %out, i64 %iv
  store i32 %value, ptr %out.ptr, align 4
  %iv.next = add nuw i64 %iv, 1
  %exit.cond = icmp eq i64 %iv.next, %n
  br i1 %exit.cond, label %exit, label %loop

exit:
  ret void
}

; Fixed-trip-count forms of the two examples from llvm.org/PR206384.

define void @issue_i32(ptr noalias readonly %conditions,
                       ptr noalias writeonly %out,
                       ptr noalias readonly %a,
                       ptr noalias readonly %b) {
; ISSUE-LABEL: @issue_i32(
; ISSUE: [[MASK:%.*]] = icmp sgt <8 x i32> {{.*}}, zeroinitializer
; ISSUE: [[NOT_MASK:%.*]] = xor <8 x i1> [[MASK]], splat (i1 true)
; ISSUE: call <8 x i32> @llvm.masked.load.v8i32.p0(ptr align 1 %a, <8 x i1> [[MASK]], <8 x i32> poison)
; ISSUE: call <8 x i32> @llvm.masked.load.v8i32.p0(ptr align 1 %b, <8 x i1> [[NOT_MASK]], <8 x i32> poison)
; ISSUE-NOT: @llvm.masked.gather
; ISSUE: ret void
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %condition.ptr = getelementptr inbounds i32, ptr %conditions, i64 %iv
  %condition = load i32, ptr %condition.ptr, align 4
  %cmp = icmp sgt i32 %condition, 0
  %base = select i1 %cmp, ptr %a, ptr %b
  %src.ptr = getelementptr inbounds i32, ptr %base, i64 %iv
  %value = load i32, ptr %src.ptr, align 4
  %out.ptr = getelementptr inbounds i32, ptr %out, i64 %iv
  store i32 %value, ptr %out.ptr, align 4
  %iv.next = add nuw i64 %iv, 1
  %exit.cond = icmp eq i64 %iv.next, 8
  br i1 %exit.cond, label %exit, label %loop

exit:
  ret void
}

define void @issue_f64(ptr noalias readonly %conditions,
                       ptr noalias writeonly %out,
                       ptr noalias readonly %a,
                       ptr noalias readonly %b) {
; ISSUE-LABEL: @issue_f64(
; ISSUE: [[MASK:%.*]] = fcmp ogt <4 x double> {{.*}}, zeroinitializer
; ISSUE: [[NOT_MASK:%.*]] = xor <4 x i1> [[MASK]], splat (i1 true)
; ISSUE: call <4 x double> @llvm.masked.load.v4f64.p0(ptr align 1 %a, <4 x i1> [[MASK]], <4 x double> poison)
; ISSUE: call <4 x double> @llvm.masked.load.v4f64.p0(ptr align 1 %b, <4 x i1> [[NOT_MASK]], <4 x double> poison)
; ISSUE-NOT: @llvm.masked.gather
; ISSUE: ret void
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %condition.ptr = getelementptr inbounds double, ptr %conditions, i64 %iv
  %condition = load double, ptr %condition.ptr, align 8
  %cmp = fcmp ogt double %condition, 0.0
  %base = select i1 %cmp, ptr %a, ptr %b
  %src.ptr = getelementptr inbounds double, ptr %base, i64 %iv
  %value = load double, ptr %src.ptr, align 8
  %out.ptr = getelementptr inbounds double, ptr %out, i64 %iv
  store double %value, ptr %out.ptr, align 8
  %iv.next = add nuw i64 %iv, 1
  %exit.cond = icmp eq i64 %iv.next, 4
  br i1 %exit.cond, label %exit, label %loop

exit:
  ret void
}

; A non-unit index is not representable by two consecutive masked loads and
; must keep the gather.

define void @selected_base_stride_two(ptr noalias readonly %conditions,
                                      ptr noalias writeonly %out,
                                      ptr noalias readonly %a,
                                      ptr noalias readonly %b, i64 %n) {
; CHECK-LABEL: @selected_base_stride_two(
; CHECK: vector.body:
; CHECK: call <8 x i32> @llvm.masked.gather.v8i32.v8p0(
; CHECK-NOT: @llvm.masked.load
; CHECK: ret void
;
; VPLAN-LABEL: VPlan for loop in 'selected_base_stride_two' after widenSelectedBaseLoads
; VPLAN: EMIT ir<%src.ptr> = getelementptr inbounds ir<%base>, ir<%index>
; VPLAN-NOT: vector-pointer
; VPLAN: EMIT-SCALAR ir<%value> = load ir<%src.ptr>
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %condition.ptr = getelementptr inbounds i32, ptr %conditions, i64 %iv
  %condition = load i32, ptr %condition.ptr, align 4
  %cmp = icmp sgt i32 %condition, 0
  %base = select i1 %cmp, ptr %a, ptr %b
  %index = shl nuw i64 %iv, 1
  %src.ptr = getelementptr inbounds i32, ptr %base, i64 %index
  %value = load i32, ptr %src.ptr, align 4
  %out.ptr = getelementptr inbounds i32, ptr %out, i64 %iv
  store i32 %value, ptr %out.ptr, align 4
  %iv.next = add nuw i64 %iv, 1
  %exit.cond = icmp eq i64 %iv.next, %n
  br i1 %exit.cond, label %exit, label %loop

exit:
  ret void
}

define void @selected_base_inbounds_cleanup(
    ptr noalias readonly dereferenceable(12) %a,
    ptr noalias readonly dereferenceable(20) %b,
    ptr noalias writeonly dereferenceable(32) %out) {
; CHECK-LABEL: @selected_base_inbounds_cleanup(
; CHECK: vector.body:
; CHECK: call <8 x i32> @llvm.masked.load
; CHECK: call <8 x i32> @llvm.masked.load
; CHECK-NOT: @llvm.masked.gather
; CHECK: ret void
;
; VPLAN-DROP-LABEL: VPlan for loop in 'selected_base_inbounds_cleanup' after VPlanTransforms::dropPoisonGeneratingRecipes
; VPLAN-DROP: CLONE ir<%src.ptr>.1 = getelementptr ir<%a.end>, ir<%index>
; VPLAN-DROP: CLONE ir<%src.ptr>.2 = getelementptr ir<%b>, ir<%index>
; VPLAN-DROP: [[A_PTR:vp<%[0-9]+>]] = vector-pointer i32, ir<%src.ptr>.1, ir<1>
; VPLAN-DROP: [[B_PTR:vp<%[0-9]+>]] = vector-pointer i32, ir<%src.ptr>.2, ir<1>

entry:
  %a.end = getelementptr inbounds i32, ptr %a, i64 3
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]

  ; iv 0..2 use a.end + (-3..-1)
  ; iv 3..7 use b + (0..4)
  %use.a = icmp ult i64 %iv, 3
  %base = select i1 %use.a, ptr %a.end, ptr %b
  %index = add i64 %iv, -3

  ; inbounds valid, only load the selected Addr
  %src.ptr = getelementptr inbounds i32, ptr %base, i64 %index
  %value = load i32, ptr %src.ptr, align 4

  %out.ptr = getelementptr inbounds i32, ptr %out, i64 %iv
  store i32 %value, ptr %out.ptr, align 4

  %iv.next = add nuw i64 %iv, 1
  %done = icmp eq i64 %iv.next, 8
  br i1 %done, label %exit, label %loop

exit:
  ret void
}