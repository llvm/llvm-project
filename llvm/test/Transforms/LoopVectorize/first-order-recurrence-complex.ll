; RUN: opt -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S %s | FileCheck %s


@p = external local_unnamed_addr global [257 x i32], align 16
@q = external local_unnamed_addr global [257 x i32], align 16

; Test case for PR43398.

define void @can_sink_after_store(i32 %x, i32* %ptr, i64 %tc) local_unnamed_addr #0 {
; CHECK-LABEL: vector.ph:
; CHECK:        %broadcast.splatinsert1 = insertelement <4 x i32> undef, i32 %x, i32 0
; CHECK-NEXT:   %broadcast.splat2 = shufflevector <4 x i32> %broadcast.splatinsert1, <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT:   %vector.recur.init = insertelement <4 x i32> undef, i32 %.pre, i32 3
; CHECK-NEXT:    br label %vector.body

; CHECK-LABEL: vector.body:
; CHECK-NEXT:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK-NEXT:   %vector.recur = phi <4 x i32> [ %vector.recur.init, %vector.ph ], [ %wide.load, %vector.body ]
; CHECK-NEXT:   %offset.idx = add i64 1, %index
; CHECK-NEXT:   %broadcast.splatinsert = insertelement <4 x i64> undef, i64 %offset.idx, i32 0
; CHECK-NEXT:   %broadcast.splat = shufflevector <4 x i64> %broadcast.splatinsert, <4 x i64> undef, <4 x i32> zeroinitializer
; CHECK-NEXT:   %induction = add <4 x i64> %broadcast.splat, <i64 0, i64 1, i64 2, i64 3>
; CHECK-NEXT:   %0 = add i64 %offset.idx, 0
; CHECK-NEXT:   %1 = getelementptr inbounds [257 x i32], [257 x i32]* @p, i64 0, i64 %0
; CHECK-NEXT:   %2 = getelementptr inbounds i32, i32* %1, i32 0
; CHECK-NEXT:   %3 = bitcast i32* %2 to <4 x i32>*
; CHECK-NEXT:   %wide.load = load <4 x i32>, <4 x i32>* %3, align 4
; CHECK-NEXT:   %4 = shufflevector <4 x i32> %vector.recur, <4 x i32> %wide.load, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:   %5 = add <4 x i32> %4, %broadcast.splat2
; CHECK-NEXT:   %6 = add <4 x i32> %5, %wide.load
; CHECK-NEXT:   %7 = getelementptr inbounds [257 x i32], [257 x i32]* @q, i64 0, i64 %0
; CHECK-NEXT:   %8 = getelementptr inbounds i32, i32* %7, i32 0
; CHECK-NEXT:   %9 = bitcast i32* %8 to <4 x i32>*
; CHECK-NEXT:   store <4 x i32> %6, <4 x i32>* %9, align 4
; CHECK-NEXT:   %index.next = add i64 %index, 4
; CHECK-NEXT:   %10 = icmp eq i64 %index.next, 1996
; CHECK-NEXT:   br i1 %10, label %middle.block, label %vector.body
;
entry:
  br label %preheader

preheader:
  %idx.phi.trans = getelementptr inbounds [257 x i32], [257 x i32]* @p, i64 0, i64 1
  %.pre = load i32, i32* %idx.phi.trans, align 4
  br label %for

for:
  %pre.phi = phi i32 [ %.pre, %preheader ], [ %pre.next, %for ]
  %iv = phi i64 [ 1, %preheader ], [ %iv.next, %for ]
  %add.1 = add i32 %pre.phi, %x
  %idx.1 = getelementptr inbounds [257 x i32], [257 x i32]* @p, i64 0, i64 %iv
  %pre.next = load i32, i32* %idx.1, align 4
  %add.2 = add i32 %add.1, %pre.next
  %idx.2 = getelementptr inbounds [257 x i32], [257 x i32]* @q, i64 0, i64 %iv
  store i32 %add.2, i32* %idx.2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 2000
  br i1 %exitcond, label %exit, label %for

exit:
  ret void
}

; We can sink potential trapping instructions, as this will only delay the trap
; and not introduce traps on additional paths.
define void @sink_sdiv(i32 %x, i32* %ptr, i64 %tc) local_unnamed_addr #0 {
; CHECK-LABEL: vector.ph:
; CHECK:        %broadcast.splatinsert1 = insertelement <4 x i32> undef, i32 %x, i32 0
; CHECK-NEXT:   %broadcast.splat2 = shufflevector <4 x i32> %broadcast.splatinsert1, <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT:   %vector.recur.init = insertelement <4 x i32> undef, i32 %.pre, i32 3
; CHECK-NEXT:    br label %vector.body

; CHECK-LABEL: vector.body:
; CHECK-NEXT:   %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK-NEXT:   %vector.recur = phi <4 x i32> [ %vector.recur.init, %vector.ph ], [ %wide.load, %vector.body ]
; CHECK-NEXT:   %offset.idx = add i64 1, %index
; CHECK-NEXT:   %broadcast.splatinsert = insertelement <4 x i64> undef, i64 %offset.idx, i32 0
; CHECK-NEXT:   %broadcast.splat = shufflevector <4 x i64> %broadcast.splatinsert, <4 x i64> undef, <4 x i32> zeroinitializer
; CHECK-NEXT:   %induction = add <4 x i64> %broadcast.splat, <i64 0, i64 1, i64 2, i64 3>
; CHECK-NEXT:   %0 = add i64 %offset.idx, 0
; CHECK-NEXT:   %1 = getelementptr inbounds [257 x i32], [257 x i32]* @p, i64 0, i64 %0
; CHECK-NEXT:   %2 = getelementptr inbounds i32, i32* %1, i32 0
; CHECK-NEXT:   %3 = bitcast i32* %2 to <4 x i32>*
; CHECK-NEXT:   %wide.load = load <4 x i32>, <4 x i32>* %3, align 4
; CHECK-NEXT:   %4 = shufflevector <4 x i32> %vector.recur, <4 x i32> %wide.load, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
; CHECK-NEXT:   %5 = sdiv <4 x i32> %4, %broadcast.splat2
; CHECK-NEXT:   %6 = add <4 x i32> %5, %wide.load
; CHECK-NEXT:   %7 = getelementptr inbounds [257 x i32], [257 x i32]* @q, i64 0, i64 %0
; CHECK-NEXT:   %8 = getelementptr inbounds i32, i32* %7, i32 0
; CHECK-NEXT:   %9 = bitcast i32* %8 to <4 x i32>*
; CHECK-NEXT:   store <4 x i32> %6, <4 x i32>* %9, align 4
; CHECK-NEXT:   %index.next = add i64 %index, 4
; CHECK-NEXT:   %10 = icmp eq i64 %index.next, 1996
; CHECK-NEXT:   br i1 %10, label %middle.block, label %vector.body
;
entry:
  br label %preheader

preheader:
  %idx.phi.trans = getelementptr inbounds [257 x i32], [257 x i32]* @p, i64 0, i64 1
  %.pre = load i32, i32* %idx.phi.trans, align 4
  br label %for

for:
  %pre.phi = phi i32 [ %.pre, %preheader ], [ %pre.next, %for ]
  %iv = phi i64 [ 1, %preheader ], [ %iv.next, %for ]
  %div.1 = sdiv i32 %pre.phi, %x
  %idx.1 = getelementptr inbounds [257 x i32], [257 x i32]* @p, i64 0, i64 %iv
  %pre.next = load i32, i32* %idx.1, align 4
  %add.2 = add i32 %div.1, %pre.next
  %idx.2 = getelementptr inbounds [257 x i32], [257 x i32]* @q, i64 0, i64 %iv
  store i32 %add.2, i32* %idx.2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 2000
  br i1 %exitcond, label %exit, label %for

exit:
  ret void
}

; FIXME: Currently we can only sink a single instruction. For the example below,
;        we also have to sink users.
define void @cannot_sink_with_additional_user(i32 %x, i32* %ptr, i64 %tc) {
; CHECK-LABEL: define void @cannot_sink_with_additional_user(
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %preheader

; CHECK-LABEL: preheader:                                        ; preds = %entry
; CHECK:  br label %for

; CHECK-LABEL: for:                                              ; preds = %for, %preheader
; CHECK  br i1 %exitcond, label %exit, label %for

; CHECK-LABEL: exit:
; CHECK-NEXT:    ret void

entry:
  br label %preheader

preheader:
  %idx.phi.trans = getelementptr inbounds [257 x i32], [257 x i32]* @p, i64 0, i64 1
  %.pre = load i32, i32* %idx.phi.trans, align 4
  br label %for

for:
  %pre.phi = phi i32 [ %.pre, %preheader ], [ %pre.next, %for ]
  %iv = phi i64 [ 1, %preheader ], [ %iv.next, %for ]
  %add.1 = add i32 %pre.phi, %x
  %add.2 = add i32 %add.1, %x
  %idx.1 = getelementptr inbounds [257 x i32], [257 x i32]* @p, i64 0, i64 %iv
  %pre.next = load i32, i32* %idx.1, align 4
  %add.3 = add i32 %add.1, %pre.next
  %add.4 = add i32 %add.2, %add.3
  %idx.2 = getelementptr inbounds [257 x i32], [257 x i32]* @q, i64 0, i64 %iv
  store i32 %add.4, i32* %idx.2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 2000
  br i1 %exitcond, label %exit, label %for

exit:
  ret void
}

; FIXME: We can sink a store, if we can guarantee that it does not alias any
;        loads/stores in between.
define void @cannot_sink_store(i32 %x, i32* %ptr, i64 %tc) {
; CHECK-LABEL: define void @cannot_sink_store(
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %preheader

; CHECK-LABEL: preheader:                                        ; preds = %entry
; CHECK:  br label %for

; CHECK-LABEL: for:                                              ; preds = %for, %preheader
; CHECK  br i1 %exitcond, label %exit, label %for

; CHECK-LABEL: exit:
; CHECK-NEXT:    ret void
;
entry:
  br label %preheader

preheader:
  %idx.phi.trans = getelementptr inbounds [257 x i32], [257 x i32]* @p, i64 0, i64 1
  %.pre = load i32, i32* %idx.phi.trans, align 4
  br label %for

for:
  %pre.phi = phi i32 [ %.pre, %preheader ], [ %pre.next, %for ]
  %iv = phi i64 [ 1, %preheader ], [ %iv.next, %for ]
  %add.1 = add i32 %pre.phi, %x
  store i32 %add.1, i32* %ptr
  %idx.1 = getelementptr inbounds [257 x i32], [257 x i32]* @p, i64 0, i64 %iv
  %pre.next = load i32, i32* %idx.1, align 4
  %add.2 = add i32 %add.1, %pre.next
  %idx.2 = getelementptr inbounds [257 x i32], [257 x i32]* @q, i64 0, i64 %iv
  store i32 %add.2, i32* %idx.2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 2000
  br i1 %exitcond, label %exit, label %for

exit:
  ret void
}

; Some kinds of reductions are not detected by IVDescriptors. If we have a
; cycle, we cannot sink it.
define void @cannot_sink_reduction(i32 %x, i32* %ptr, i64 %tc) {
; CHECK-LABEL: define void @cannot_sink_reduction(
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %preheader

; CHECK-LABEL: preheader:                                        ; preds = %entry
; CHECK:  br label %for

; CHECK-LABEL: for:                                              ; preds = %for, %preheader
; CHECK  br i1 %exitcond, label %exit, label %for

; CHECK-LABEL: exit:                                    ; preds = %for
; CHECK-NET:     ret void
;
entry:
  br label %preheader

preheader:
  %idx.phi.trans = getelementptr inbounds [257 x i32], [257 x i32]* @p, i64 0, i64 1
  %.pre = load i32, i32* %idx.phi.trans, align 4
  br label %for

for:
  %pre.phi = phi i32 [ %.pre, %preheader ], [ %d, %for ]
  %iv = phi i64 [ 1, %preheader ], [ %iv.next, %for ]
  %d = sdiv i32 %pre.phi, %x
  %idx.1 = getelementptr inbounds [257 x i32], [257 x i32]* @p, i64 0, i64 %iv
  %pre.next = load i32, i32* %idx.1, align 4
  %add.2 = add i32 %x, %pre.next
  %idx.2 = getelementptr inbounds [257 x i32], [257 x i32]* @q, i64 0, i64 %iv
  store i32 %add.2, i32* %idx.2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 2000
  br i1 %exitcond, label %exit, label %for

exit:
  ret void
}

; TODO: We should be able to sink %tmp38 after %tmp60.
define void @instruction_with_2_FOR_operands() {
; CHECK-LABEL: define void @instruction_with_2_FOR_operands(
; CHECK-NEXT: bb:
; CHECK-NEXT:   br label %bb13

; CHECK-LABEL: bb13:
; CHECK:         br i1 %tmp12, label %bb13, label %bb74

; CHECK-LABEL: bb74:
; CHECK-NEXT:    ret void
;
bb:
  br label %bb13

bb13:                                             ; preds = %bb13, %bb
  %tmp37 = phi float [ %tmp60, %bb13 ], [ undef, %bb ]
  %tmp27 = phi float [ %tmp49, %bb13 ], [ undef, %bb ]
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb13 ], [ 0, %bb ]
  %tmp38 = fmul fast float %tmp37, %tmp27
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %tmp49 = load float, float* undef, align 4
  %tmp60 = load float, float* undef, align 4
  %tmp12 = icmp slt i64 %indvars.iv, undef
  br i1 %tmp12, label %bb13, label %bb74

bb74:                                             ; preds = %bb13
  ret void
}
