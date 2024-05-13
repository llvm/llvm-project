; RUN: opt -mtriple=thumbv8.1m.main-none-none-eabi -mattr=+mve.fp < %s -codegenprepare -S | FileCheck -check-prefix=CHECK %s

define void @sink_add_mul(ptr %s1, i32 %x, ptr %d, i32 %n) {
; CHECK-LABEL: @sink_add_mul(
; CHECK:    vector.ph:
; CHECK-NOT:  %{{.*}} = insertelement <4 x i32> undef, i32 %{{.*}}, i32 0
; CHECK-NOT:  %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> undef, <4 x i32> zeroinitializer
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
  %0 = getelementptr inbounds i32, ptr %s1, i32 %index
  %wide.load = load <4 x i32>, ptr %0, align 4
  %1 = mul nsw <4 x i32> %wide.load, %broadcast.splat9
  %2 = getelementptr inbounds i32, ptr %d, i32 %index
  %wide.load10 = load <4 x i32>, ptr %2, align 4
  %3 = add nsw <4 x i32> %wide.load10, %1
  store <4 x i32> %3, ptr %2, align 4
  %index.next = add i32 %index, 4
  %4 = icmp eq i32 %index.next, %n.vec
  br i1 %4, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void
}

define void @sink_add_mul_multiple(ptr %s1, ptr %s2, i32 %x, ptr %d, ptr %d2, i32 %n) {
; CHECK-LABEL: @sink_add_mul_multiple(
; CHECK:    vector.ph:
; CHECK-NOT:  %{{.*}} = insertelement <4 x i32> undef, i32 %{{.*}}, i32 0
; CHECK-NOT:  %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> undef, <4 x i32> zeroinitializer
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
  %0 = getelementptr inbounds i32, ptr %s1, i32 %index
  %wide.load = load <4 x i32>, ptr %0, align 4
  %1 = mul nsw <4 x i32> %wide.load, %broadcast.splat16
  %2 = getelementptr inbounds i32, ptr %d, i32 %index
  %wide.load17 = load <4 x i32>, ptr %2, align 4
  %3 = add nsw <4 x i32> %wide.load17, %1
  store <4 x i32> %3, ptr %2, align 4
  %4 = getelementptr inbounds i32, ptr %s2, i32 %index
  %wide.load18 = load <4 x i32>, ptr %4, align 4
  %5 = mul nsw <4 x i32> %wide.load18, %broadcast.splat16
  %6 = getelementptr inbounds i32, ptr %d2, i32 %index
  %wide.load19 = load <4 x i32>, ptr %6, align 4
  %7 = add nsw <4 x i32> %wide.load19, %5
  store <4 x i32> %7, ptr %6, align 4
  %index.next = add i32 %index, 4
  %8 = icmp eq i32 %index.next, %n.vec
  br i1 %8, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void
}


define void @sink_add_sub_unsinkable(ptr %s1, ptr %s2, i32 %x, ptr %d, ptr %d2, i32 %n) {
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
  %0 = getelementptr inbounds i32, ptr %s1, i32 %index
  %wide.load = load <4 x i32>, ptr %0, align 4
  %1 = mul nsw <4 x i32> %wide.load, %broadcast.splat16
  %2 = getelementptr inbounds i32, ptr %d, i32 %index
  %wide.load17 = load <4 x i32>, ptr %2, align 4
  %3 = add nsw <4 x i32> %wide.load17, %1
  store <4 x i32> %3, ptr %2, align 4
  %4 = getelementptr inbounds i32, ptr %s2, i32 %index
  %wide.load18 = load <4 x i32>, ptr %4, align 4
  %5 = sub nsw <4 x i32> %broadcast.splat16, %wide.load18
  %6 = getelementptr inbounds i32, ptr %d2, i32 %index
  %wide.load19 = load <4 x i32>, ptr %6, align 4
  %7 = add nsw <4 x i32> %wide.load19, %5
  store <4 x i32> %7, ptr %6, align 4
  %index.next = add i32 %index, 4
  %8 = icmp eq i32 %index.next, %n.vec
  br i1 %8, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void
}

define void @sink_sub(ptr %s1, i32 %x, ptr %d, i32 %n) {
; CHECK-LABEL: @sink_sub(
; CHECK:    vector.ph:
; CHECK-NOT:  %{{.*}} = insertelement <4 x i32> undef, i32 %{{.*}}, i32 0
; CHECK-NOT:  %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> undef, <4 x i32> zeroinitializer
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
  %0 = getelementptr inbounds i32, ptr %s1, i32 %index
  %wide.load = load <4 x i32>, ptr %0, align 4
  %1 = sub nsw <4 x i32> %wide.load, %broadcast.splat9
  %2 = getelementptr inbounds i32, ptr %d, i32 %index
  store <4 x i32> %1, ptr %2, align 4
  %index.next = add i32 %index, 4
  %3 = icmp eq i32 %index.next, %n.vec
  br i1 %3, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void
}

define void @sink_sub_unsinkable(ptr %s1, i32 %x, ptr %d, i32 %n) {
entry:
; CHECK-LABEL: @sink_sub_unsinkable(
; CHECK:      vector.ph:
; CHECK-NEXT:   [[N_VEC:%.*]] = and i32 [[N]], -4
; CHECK-NEXT:   [[BROADCAST_SPLATINSERT15:%.*]] = insertelement <4 x i32> undef, i32 [[X:%.*]], i32 0
; CHECK-NEXT:   [[BROADCAST_SPLAT16:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT15]], <4 x i32> undef, <4 x i32> zeroinitializer
; CHECK-NEXT:   br label [[VECTOR_BODY:%.*]]
; CHECK:      vector.body:
; CHECK-NOT:    %{{.*}} = insertelement <4 x i32> undef, i32 %{{.*}}, i32 0
; CHECK-NOT:    %{{.*}} = shufflevector <4 x i32> %{{.*}}, <4 x i32> undef, <4 x i32> zeroinitializer
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
  %0 = getelementptr inbounds i32, ptr %s1, i32 %index
  %wide.load = load <4 x i32>, ptr %0, align 4
  %1 = sub nsw <4 x i32> %broadcast.splat9, %wide.load
  %2 = getelementptr inbounds i32, ptr %d, i32 %index
  store <4 x i32> %1, ptr %2, align 4
  %index.next = add i32 %index, 4
  %3 = icmp eq i32 %index.next, %n.vec
  br i1 %3, label %for.cond.cleanup, label %vector.body

for.cond.cleanup:                                 ; preds = %for.body, %middle.block, %entry
  ret void
}
