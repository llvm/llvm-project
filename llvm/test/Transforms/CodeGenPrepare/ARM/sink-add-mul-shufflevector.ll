; RUN: opt -mtriple=thumbv8.1m.main-arm-none-eabi -mattr=+mve.fp < %s -codegenprepare -S | FileCheck -check-prefix=CHECK %s

define void @sink_add_mul(i32* %s1, i32 %x, i32* %d, i32 %n) {
; CHECK-LABEL: @sink_add_mul(
; CHECK:    vector.ph:
; CHECK-NOT:  [[BROADCAST_SPLATINSERT8:%.*]] = insertelement <4 x i32> undef, i32 [[X:%.*]], i32 0
; CHECK-NOT:  [[BROADCAST_SPLAT9:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT8]], <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK:    vector.body:
; CHECK:      [[TMP2:%.*]] = insertelement <4 x i32> undef, i32 [[X:%.*]], i32 0
; CHECK:      [[TMP3:%.*]] = shufflevector <4 x i32> [[TMP2]], <4 x i32> undef, <4 x i32> zeroinitializer
;
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %vector.ph, label %for.cond.cleanup

vector.ph:                                        ; preds = %for.body.preheader
  %n.vec = and i32 %n, -4
  %broadcast.splatinsert8 = insertelement <4 x i32> undef, i32 %x, i32 0
  %broadcast.splat9 = shufflevector <4 x i32> %broadcast.splatinsert8, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i32, i32* %s1, i32 %index
  %1 = bitcast i32* %0 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %1, align 4
  %2 = mul nsw <4 x i32> %wide.load, %broadcast.splat9
  %3 = getelementptr inbounds i32, i32* %d, i32 %index
  %4 = bitcast i32* %3 to <4 x i32>*
  %wide.load10 = load <4 x i32>, <4 x i32>* %4, align 4
  %5 = add nsw <4 x i32> %wide.load10, %2
  %6 = bitcast i32* %3 to <4 x i32>*
  store <4 x i32> %5, <4 x i32>* %6, align 4
  %index.next = add i32 %index, 4
  %7 = icmp eq i32 %index.next, %n.vec
  br i1 %7, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void
}

define void @sink_add_mul_multiple(i32* %s1, i32* %s2, i32 %x, i32* %d, i32* %d2, i32 %n) {
; CHECK-LABEL: @sink_add_mul_multiple(
; CHECK:    vector.ph:
; CHECK-NOT:  [[BROADCAST_SPLATINSERT8:%.*]] = insertelement <4 x i32> undef, i32 [[X:%.*]], i32 0
; CHECK-NOT:  [[BROADCAST_SPLAT9:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT8]], <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK:    vector.body:
; CHECK:      [[TMP2:%.*]] = insertelement <4 x i32> undef, i32 %x, i32 0
; CHECK:      [[TMP3:%.*]] = shufflevector <4 x i32> [[TMP2]], <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK:      mul nsw <4 x i32> %wide.load, [[TMP3]]
; CHECK:      [[TMP2b:%.*]] = insertelement <4 x i32> undef, i32 %x, i32 0
; CHECK:      [[TMP3b:%.*]] = shufflevector <4 x i32> [[TMP2b]], <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK:      mul nsw <4 x i32> %wide.load18, [[TMP3b]]
;
entry:
  %cmp13 = icmp sgt i32 %n, 0
  br i1 %cmp13, label %vector.ph, label %for.cond.cleanup

vector.ph:                                        ; preds = %for.body.preheader
  %n.vec = and i32 %n, -4
  %broadcast.splatinsert15 = insertelement <4 x i32> undef, i32 %x, i32 0
  %broadcast.splat16 = shufflevector <4 x i32> %broadcast.splatinsert15, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i32, i32* %s1, i32 %index
  %1 = bitcast i32* %0 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %1, align 4
  %2 = mul nsw <4 x i32> %wide.load, %broadcast.splat16
  %3 = getelementptr inbounds i32, i32* %d, i32 %index
  %4 = bitcast i32* %3 to <4 x i32>*
  %wide.load17 = load <4 x i32>, <4 x i32>* %4, align 4
  %5 = add nsw <4 x i32> %wide.load17, %2
  %6 = bitcast i32* %3 to <4 x i32>*
  store <4 x i32> %5, <4 x i32>* %6, align 4
  %7 = getelementptr inbounds i32, i32* %s2, i32 %index
  %8 = bitcast i32* %7 to <4 x i32>*
  %wide.load18 = load <4 x i32>, <4 x i32>* %8, align 4
  %9 = mul nsw <4 x i32> %wide.load18, %broadcast.splat16
  %10 = getelementptr inbounds i32, i32* %d2, i32 %index
  %11 = bitcast i32* %10 to <4 x i32>*
  %wide.load19 = load <4 x i32>, <4 x i32>* %11, align 4
  %12 = add nsw <4 x i32> %wide.load19, %9
  %13 = bitcast i32* %10 to <4 x i32>*
  store <4 x i32> %12, <4 x i32>* %13, align 4
  %index.next = add i32 %index, 4
  %14 = icmp eq i32 %index.next, %n.vec
  br i1 %14, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void
}


define void @sink_add_sub_unsinkable(i32* %s1, i32* %s2, i32 %x, i32* %d, i32* %d2, i32 %n) {
; CHECK-LABEL: @sink_add_sub_unsinkable(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP13:%.*]] = icmp sgt i32 [[N:%.*]], 0
; CHECK-NEXT:    br i1 [[CMP13]], label [[VECTOR_PH:%.*]], label [[FOR_COND_CLEANUP:%.*]]
; CHECK:       vector.ph:
; CHECK-NEXT:    [[N_VEC:%.*]] = and i32 [[N]], -4
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT15:%.*]] = insertelement <4 x i32> undef, i32 [[X:%.*]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT16:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT15]], <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label [[VECTOR_BODY:%.*]]
;
entry:
  %cmp13 = icmp sgt i32 %n, 0
  br i1 %cmp13, label %vector.ph, label %for.cond.cleanup

vector.ph:                                        ; preds = %for.body.preheader
  %n.vec = and i32 %n, -4
  %broadcast.splatinsert15 = insertelement <4 x i32> undef, i32 %x, i32 0
  %broadcast.splat16 = shufflevector <4 x i32> %broadcast.splatinsert15, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i32, i32* %s1, i32 %index
  %1 = bitcast i32* %0 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %1, align 4
  %2 = mul nsw <4 x i32> %wide.load, %broadcast.splat16
  %3 = getelementptr inbounds i32, i32* %d, i32 %index
  %4 = bitcast i32* %3 to <4 x i32>*
  %wide.load17 = load <4 x i32>, <4 x i32>* %4, align 4
  %5 = add nsw <4 x i32> %wide.load17, %2
  %6 = bitcast i32* %3 to <4 x i32>*
  store <4 x i32> %5, <4 x i32>* %6, align 4
  %7 = getelementptr inbounds i32, i32* %s2, i32 %index
  %8 = bitcast i32* %7 to <4 x i32>*
  %wide.load18 = load <4 x i32>, <4 x i32>* %8, align 4
  %9 = sub nsw <4 x i32> %broadcast.splat16, %wide.load18
  %10 = getelementptr inbounds i32, i32* %d2, i32 %index
  %11 = bitcast i32* %10 to <4 x i32>*
  %wide.load19 = load <4 x i32>, <4 x i32>* %11, align 4
  %12 = add nsw <4 x i32> %wide.load19, %9
  %13 = bitcast i32* %10 to <4 x i32>*
  store <4 x i32> %12, <4 x i32>* %13, align 4
  %index.next = add i32 %index, 4
  %14 = icmp eq i32 %index.next, %n.vec
  br i1 %14, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void
}

define void @sink_sub(i32* %s1, i32 %x, i32* %d, i32 %n) {
; CHECK-LABEL: @sink_sub(
; CHECK:    vector.ph:
; CHECK-NOT:  [[BROADCAST_SPLATINSERT8:%.*]] = insertelement <4 x i32> undef, i32 [[X:%.*]], i32 0
; CHECK-NOT:  [[BROADCAST_SPLAT9:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT8]], <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK:    vector.body:
; CHECK:      [[TMP2:%.*]] = insertelement <4 x i32> undef, i32 [[X:%.*]], i32 0
; CHECK:      [[TMP3:%.*]] = shufflevector <4 x i32> [[TMP2]], <4 x i32> undef, <4 x i32> zeroinitializer
;
entry:
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %vector.ph, label %for.cond.cleanup

vector.ph:                                        ; preds = %for.body.preheader
  %n.vec = and i32 %n, -4
  %broadcast.splatinsert8 = insertelement <4 x i32> undef, i32 %x, i32 0
  %broadcast.splat9 = shufflevector <4 x i32> %broadcast.splatinsert8, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i32, i32* %s1, i32 %index
  %1 = bitcast i32* %0 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %1, align 4
  %2 = sub nsw <4 x i32> %wide.load, %broadcast.splat9
  %3 = getelementptr inbounds i32, i32* %d, i32 %index
  %4 = bitcast i32* %3 to <4 x i32>*
  store <4 x i32> %2, <4 x i32>* %4, align 4
  %index.next = add i32 %index, 4
  %5 = icmp eq i32 %index.next, %n.vec
  br i1 %5, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void
}

define void @sink_sub_unsinkable(i32* %s1, i32 %x, i32* %d, i32 %n) {
entry:
; CHECK-LABEL: @sink_sub_unsinkable(
; CHECK:      vector.ph:
; CHECK-NEXT:   [[N_VEC:%.*]] = and i32 [[N]], -4
; CHECK-NEXT:   [[BROADCAST_SPLATINSERT15:%.*]] = insertelement <4 x i32> undef, i32 [[X:%.*]], i32 0
; CHECK-NEXT:   [[BROADCAST_SPLAT16:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT15]], <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT:   br label [[VECTOR_BODY:%.*]]
; CHECK:      vector.body:
; CHECK-NOT:    [[TMP2:%.*]] = insertelement <4 x i32> undef, i32 [[X:%.*]], i32 0
; CHECK-NOT:    [[TMP3:%.*]] = shufflevector <4 x i32> [[TMP2]], <4 x i32> undef, <4 x i32> zeroinitializer
;
  %cmp6 = icmp sgt i32 %n, 0
  br i1 %cmp6, label %vector.ph, label %for.cond.cleanup

vector.ph:                                        ; preds = %for.body.preheader
  %n.vec = and i32 %n, -4
  %broadcast.splatinsert8 = insertelement <4 x i32> undef, i32 %x, i32 0
  %broadcast.splat9 = shufflevector <4 x i32> %broadcast.splatinsert8, <4 x i32> undef, <4 x i32> zeroinitializer
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %0 = getelementptr inbounds i32, i32* %s1, i32 %index
  %1 = bitcast i32* %0 to <4 x i32>*
  %wide.load = load <4 x i32>, <4 x i32>* %1, align 4
  %2 = sub nsw <4 x i32> %broadcast.splat9, %wide.load
  %3 = getelementptr inbounds i32, i32* %d, i32 %index
  %4 = bitcast i32* %3 to <4 x i32>*
  store <4 x i32> %2, <4 x i32>* %4, align 4
  %index.next = add i32 %index, 4
  %5 = icmp eq i32 %index.next, %n.vec
  br i1 %5, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void
}
