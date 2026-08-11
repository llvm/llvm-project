; RUN: opt -passes='loop-mssa(indvars)' -verify-scev -S %s | FileCheck %s

; Two pointer IVs in a single loop, 4 bytes apart.
define void @single_loop(ptr noalias %dst, i32 %n) {
; CHECK-LABEL: define void @single_loop(
; CHECK-SAME: ptr noalias [[DST:%.*]], i32 [[N:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[PA:%.*]] = phi ptr [ [[DST]], %[[ENTRY]] ], [ [[PA_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[I_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[PB_OFF:%.*]] = getelementptr i8, ptr [[PA]], i64 4
; CHECK-NEXT:    store float 2.000000e+00, ptr [[PA]], align 4
; CHECK-NEXT:    store float 3.000000e+00, ptr [[PB_OFF]], align 4
; CHECK-NEXT:    [[PA_NEXT]] = getelementptr inbounds float, ptr [[PA]], i64 2
; CHECK-NEXT:    [[I_NEXT]] = add nuw i32 [[I]], 1
; CHECK-NEXT:    [[CMP:%.*]] = icmp ult i32 [[I_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[CMP]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
entry:
  %pB.start = getelementptr inbounds float, ptr %dst, i64 1
  br label %loop

loop:
  %pA = phi ptr [ %dst, %entry ], [ %pA.next, %loop ]
  %pB = phi ptr [ %pB.start, %entry ], [ %pB.next, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  store float 2.000000e+00, ptr %pA, align 4
  store float 3.000000e+00, ptr %pB, align 4
  %pA.next = getelementptr inbounds float, ptr %pA, i64 2
  %pB.next = getelementptr inbounds float, ptr %pB, i64 2
  %i.next = add nuw i32 %i, 1
  %cmp = icmp ult i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

define void @nested_loop(ptr noalias %dst, i32 %width, i32 %height) {
; CHECK-LABEL: define void @nested_loop(
; CHECK-SAME: ptr noalias [[DST:%.*]], i32 [[WIDTH:%.*]], i32 [[HEIGHT:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[OUTER_HEADER:.*]]
; CHECK:       [[OUTER_HEADER]]:
; CHECK-NEXT:    [[PA_OUTER:%.*]] = phi ptr [ [[DST]], %[[ENTRY]] ], [ [[PA_LCSSA:%.*]], %[[OUTER_LATCH:.*]] ]
; CHECK-NEXT:    [[Y:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[Y_NEXT:%.*]], %[[OUTER_LATCH]] ]
; CHECK-NEXT:    [[OUTER_CMP:%.*]] = icmp ult i32 [[Y]], [[HEIGHT]]
; CHECK-NEXT:    br i1 [[OUTER_CMP]], label %[[INNER_PREHEADER:.*]], label %[[EXIT:.*]]
; CHECK:       [[INNER_PREHEADER]]:
; CHECK-NEXT:    br label %[[INNER_HEADER:.*]]
; CHECK:       [[INNER_HEADER]]:
; CHECK-NEXT:    [[PA_INNER:%.*]] = phi ptr [ [[PA_OUTER]], %[[INNER_PREHEADER]] ], [ [[PA_NEXT:%.*]], %[[INNER_HEADER]] ]
; CHECK-NEXT:    [[X:%.*]] = phi i32 [ 0, %[[INNER_PREHEADER]] ], [ [[X_NEXT:%.*]], %[[INNER_HEADER]] ]
; CHECK-NEXT:    [[PB_INNER_OFF:%.*]] = getelementptr i8, ptr [[PA_INNER]], i64 4
; CHECK-NEXT:    store float 2.000000e+00, ptr [[PA_INNER]], align 4
; CHECK-NEXT:    store float 3.000000e+00, ptr [[PB_INNER_OFF]], align 4
; CHECK-NEXT:    [[PA_NEXT]] = getelementptr inbounds float, ptr [[PA_INNER]], i64 2
; CHECK-NEXT:    [[X_NEXT]] = add nuw i32 [[X]], 1
; CHECK-NEXT:    [[INNER_CMP:%.*]] = icmp ult i32 [[X_NEXT]], [[WIDTH]]
; CHECK-NEXT:    br i1 [[INNER_CMP]], label %[[INNER_HEADER]], label %[[INNER_EXIT:.*]]
; CHECK:       [[INNER_EXIT]]:
; CHECK-NEXT:    [[PA_LCSSA]] = phi ptr [ [[PA_NEXT]], %[[INNER_HEADER]] ]
; CHECK-NEXT:    br label %[[OUTER_LATCH]]
; CHECK:       [[OUTER_LATCH]]:
; CHECK-NEXT:    [[Y_NEXT]] = add nuw i32 [[Y]], 1
; CHECK-NEXT:    br label %[[OUTER_HEADER]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
entry:
  %pB.start = getelementptr inbounds float, ptr %dst, i64 1
  br label %outer.header

outer.header:
  %pA.outer = phi ptr [ %dst, %entry ], [ %pA.lcssa, %outer.latch ]
  %pB.outer = phi ptr [ %pB.start, %entry ], [ %pB.lcssa, %outer.latch ]
  %y = phi i32 [ 0, %entry ], [ %y.next, %outer.latch ]
  %outer.cmp = icmp ult i32 %y, %height
  br i1 %outer.cmp, label %inner.preheader, label %exit

inner.preheader:
  br label %inner.header

inner.header:
  %pA.inner = phi ptr [ %pA.outer, %inner.preheader ], [ %pA.next, %inner.header ]
  %pB.inner = phi ptr [ %pB.outer, %inner.preheader ], [ %pB.next, %inner.header ]
  %x = phi i32 [ 0, %inner.preheader ], [ %x.next, %inner.header ]
  store float 2.000000e+00, ptr %pA.inner, align 4
  store float 3.000000e+00, ptr %pB.inner, align 4
  %pA.next = getelementptr inbounds float, ptr %pA.inner, i64 2
  %pB.next = getelementptr inbounds float, ptr %pB.inner, i64 2
  %x.next = add nuw i32 %x, 1
  %inner.cmp = icmp ult i32 %x.next, %width
  br i1 %inner.cmp, label %inner.header, label %inner.exit

inner.exit:
  %pA.lcssa = phi ptr [ %pA.next, %inner.header ]
  %pB.lcssa = phi ptr [ %pB.next, %inner.header ]
  br label %outer.latch

outer.latch:
  %y.next = add nuw i32 %y, 1
  br label %outer.header

exit:
  ret void
}

; Interleave factor 3: all three collapse onto the lowest-addressed IV.
define void @three_ptr_ivs(ptr noalias %dst, i32 %n) {
; CHECK-LABEL: define void @three_ptr_ivs(
; CHECK-SAME: ptr noalias [[DST:%.*]], i32 [[N:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[PA:%.*]] = phi ptr [ [[DST]], %[[ENTRY]] ], [ [[PA_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[I_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[PB_OFF:%.*]] = getelementptr i8, ptr [[PA]], i64 4
; CHECK-NEXT:    [[PC_OFF:%.*]] = getelementptr i8, ptr [[PA]], i64 8
; CHECK-NEXT:    store float 2.000000e+00, ptr [[PA]], align 4
; CHECK-NEXT:    store float 3.000000e+00, ptr [[PB_OFF]], align 4
; CHECK-NEXT:    store float 4.000000e+00, ptr [[PC_OFF]], align 4
; CHECK-NEXT:    [[PA_NEXT]] = getelementptr inbounds float, ptr [[PA]], i64 3
; CHECK-NEXT:    [[I_NEXT]] = add nuw i32 [[I]], 1
; CHECK-NEXT:    [[CMP:%.*]] = icmp ult i32 [[I_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[CMP]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
entry:
  %pB.start = getelementptr inbounds float, ptr %dst, i64 1
  %pC.start = getelementptr inbounds float, ptr %dst, i64 2
  br label %loop

loop:
  %pA = phi ptr [ %dst, %entry ], [ %pA.next, %loop ]
  %pB = phi ptr [ %pB.start, %entry ], [ %pB.next, %loop ]
  %pC = phi ptr [ %pC.start, %entry ], [ %pC.next, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  store float 2.000000e+00, ptr %pA, align 4
  store float 3.000000e+00, ptr %pB, align 4
  store float 4.000000e+00, ptr %pC, align 4
  %pA.next = getelementptr inbounds float, ptr %pA, i64 3
  %pB.next = getelementptr inbounds float, ptr %pB, i64 3
  %pC.next = getelementptr inbounds float, ptr %pC, i64 3
  %i.next = add nuw i32 %i, 1
  %cmp = icmp ult i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; The lowest-addressed IV wins even when it is not the first phi in the header,
; so every emitted GEP has a non-negative offset.
define void @keeper_is_not_first_phi(ptr noalias %dst, i32 %n) {
; CHECK-LABEL: define void @keeper_is_not_first_phi(
; CHECK-SAME: ptr noalias [[DST:%.*]], i32 [[N:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[LOOP:.*]]
; CHECK:       [[LOOP]]:
; CHECK-NEXT:    [[PLO:%.*]] = phi ptr [ [[DST]], %[[ENTRY]] ], [ [[PLO_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[I:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[I_NEXT:%.*]], %[[LOOP]] ]
; CHECK-NEXT:    [[PHI_OFF:%.*]] = getelementptr i8, ptr [[PLO]], i64 4
; CHECK-NEXT:    store float 2.000000e+00, ptr [[PHI_OFF]], align 4
; CHECK-NEXT:    store float 3.000000e+00, ptr [[PLO]], align 4
; CHECK-NEXT:    [[PLO_NEXT]] = getelementptr inbounds float, ptr [[PLO]], i64 2
; CHECK-NEXT:    [[I_NEXT]] = add nuw i32 [[I]], 1
; CHECK-NEXT:    [[CMP:%.*]] = icmp ult i32 [[I_NEXT]], [[N]]
; CHECK-NEXT:    br i1 [[CMP]], label %[[LOOP]], label %[[EXIT:.*]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
entry:
  %pHi.start = getelementptr inbounds float, ptr %dst, i64 1
  br label %loop

loop:
  %pHi = phi ptr [ %pHi.start, %entry ], [ %pHi.next, %loop ]
  %pLo = phi ptr [ %dst, %entry ], [ %pLo.next, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  store float 2.000000e+00, ptr %pHi, align 4
  store float 3.000000e+00, ptr %pLo, align 4
  %pHi.next = getelementptr inbounds float, ptr %pHi, i64 2
  %pLo.next = getelementptr inbounds float, ptr %pLo, i64 2
  %i.next = add nuw i32 %i, 1
  %cmp = icmp ult i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

; Same nest, but not rotated: the inner loop has a separate latch and the LCSSA
; phis forward the inner header phis rather than the increments.
define void @nested_loop_unrotated(ptr noalias %dst, i32 %width, i32 %height) {
; CHECK-LABEL: define void @nested_loop_unrotated(
; CHECK-SAME: ptr noalias [[DST:%.*]], i32 [[WIDTH:%.*]], i32 [[HEIGHT:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*]]:
; CHECK-NEXT:    br label %[[OUTER_HEADER:.*]]
; CHECK:       [[OUTER_HEADER]]:
; CHECK-NEXT:    [[PA_OUTER:%.*]] = phi ptr [ [[DST]], %[[ENTRY]] ], [ [[PA_LCSSA:%.*]], %[[OUTER_LATCH:.*]] ]
; CHECK-NEXT:    [[Y:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ [[Y_NEXT:%.*]], %[[OUTER_LATCH]] ]
; CHECK-NEXT:    [[OUTER_CMP:%.*]] = icmp ult i32 [[Y]], [[HEIGHT]]
; CHECK-NEXT:    br i1 [[OUTER_CMP]], label %[[INNER_PREHEADER:.*]], label %[[EXIT:.*]]
; CHECK:       [[INNER_PREHEADER]]:
; CHECK-NEXT:    br label %[[INNER_HEADER:.*]]
; CHECK:       [[INNER_HEADER]]:
; CHECK-NEXT:    [[PA_INNER:%.*]] = phi ptr [ [[PA_OUTER]], %[[INNER_PREHEADER]] ], [ [[PA_NEXT:%.*]], %[[INNER_LATCH:.*]] ]
; CHECK-NEXT:    [[X:%.*]] = phi i32 [ 0, %[[INNER_PREHEADER]] ], [ [[X_NEXT:%.*]], %[[INNER_LATCH]] ]
; CHECK-NEXT:    [[PB_INNER_OFF:%.*]] = getelementptr i8, ptr [[PA_INNER]], i64 4
; CHECK-NEXT:    [[INNER_CMP:%.*]] = icmp ult i32 [[X]], [[WIDTH]]
; CHECK-NEXT:    br i1 [[INNER_CMP]], label %[[INNER_BODY:.*]], label %[[INNER_EXIT:.*]]
; CHECK:       [[INNER_BODY]]:
; CHECK-NEXT:    store float 2.000000e+00, ptr [[PA_INNER]], align 4
; CHECK-NEXT:    store float 3.000000e+00, ptr [[PB_INNER_OFF]], align 4
; CHECK-NEXT:    [[PA_NEXT]] = getelementptr inbounds float, ptr [[PA_INNER]], i64 2
; CHECK-NEXT:    br label %[[INNER_LATCH]]
; CHECK:       [[INNER_LATCH]]:
; CHECK-NEXT:    [[X_NEXT]] = add nuw i32 [[X]], 1
; CHECK-NEXT:    br label %[[INNER_HEADER]]
; CHECK:       [[INNER_EXIT]]:
; CHECK-NEXT:    [[PA_LCSSA]] = phi ptr [ [[PA_INNER]], %[[INNER_HEADER]] ]
; CHECK-NEXT:    br label %[[OUTER_LATCH]]
; CHECK:       [[OUTER_LATCH]]:
; CHECK-NEXT:    [[Y_NEXT]] = add nuw i32 [[Y]], 1
; CHECK-NEXT:    br label %[[OUTER_HEADER]]
; CHECK:       [[EXIT]]:
; CHECK-NEXT:    ret void
;
entry:
  %pB.start = getelementptr inbounds float, ptr %dst, i64 1
  br label %outer.header

outer.header:
  %pA.outer = phi ptr [ %dst, %entry ], [ %pA.lcssa, %outer.latch ]
  %pB.outer = phi ptr [ %pB.start, %entry ], [ %pB.lcssa, %outer.latch ]
  %y = phi i32 [ 0, %entry ], [ %y.next, %outer.latch ]
  %outer.cmp = icmp ult i32 %y, %height
  br i1 %outer.cmp, label %inner.preheader, label %exit

inner.preheader:
  br label %inner.header

inner.header:
  %pA.inner = phi ptr [ %pA.outer, %inner.preheader ], [ %pA.next, %inner.latch ]
  %pB.inner = phi ptr [ %pB.outer, %inner.preheader ], [ %pB.next, %inner.latch ]
  %x = phi i32 [ 0, %inner.preheader ], [ %x.next, %inner.latch ]
  %inner.cmp = icmp ult i32 %x, %width
  br i1 %inner.cmp, label %inner.body, label %inner.exit

inner.body:
  store float 2.000000e+00, ptr %pA.inner, align 4
  store float 3.000000e+00, ptr %pB.inner, align 4
  %pA.next = getelementptr inbounds float, ptr %pA.inner, i64 2
  %pB.next = getelementptr inbounds float, ptr %pB.inner, i64 2
  br label %inner.latch

inner.latch:
  %x.next = add nuw i32 %x, 1
  br label %inner.header

inner.exit:
  %pA.lcssa = phi ptr [ %pA.inner, %inner.header ]
  %pB.lcssa = phi ptr [ %pB.inner, %inner.header ]
  br label %outer.latch

outer.latch:
  %y.next = add nuw i32 %y, 1
  br label %outer.header

exit:
  ret void
}
