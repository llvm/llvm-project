; RUN: opt -loop-vectorize -force-vector-width=4 -S %s | FileCheck %s

@dst = external global [32 x i16], align 1

define void @blend_uniform_iv_trunc(i1 %c) {
; CHECK-LABEL: @blend_uniform_iv_trunc(
; CHECK:       vector.ph:
; CHECK-NEXT:    [[MASK0:%.*]] = insertelement <4 x i1> poison, i1 %c, i32 0
; CHECK-NEXT:    [[MASK1:%.*]] = shufflevector <4 x i1> [[MASK0]], <4 x i1> poison, <4 x i32> zeroinitializer

; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = trunc i64 [[INDEX]] to i16
; CHECK-NEXT:    [[TMP2:%.*]] = add i16 [[TMP1]], 0
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <4 x i16> poison, i16 [[TMP2]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <4 x i16> [[BROADCAST_SPLATINSERT1]], <4 x i16> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP3:%.*]] = xor <4 x i1> [[MASK1]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <4 x i1> [[MASK1]], <4 x i16> [[BROADCAST_SPLAT2]], <4 x i16> undef
; CHECK-NEXT:    [[TMP4:%.*]] = extractelement <4 x i16> [[PREDPHI]], i32 0
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds [32 x i16], [32 x i16]* @dst, i16 0, i16 [[TMP4]]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds i16, i16* [[TMP5]], i32 0
; CHECK-NEXT:    [[TMP7:%.*]] = bitcast i16* [[TMP6]] to <4 x i16>*
; CHECK-NEXT:    store <4 x i16> zeroinitializer, <4 x i16>* [[TMP7]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP8:%.*]] = icmp eq i64 [[INDEX_NEXT]], 32
; CHECK-NEXT:    br i1 [[TMP8]], label %middle.block, label %vector.body
;
entry:
  br label %loop.header

loop.header:                                      ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %iv.trunc.2 = trunc i64 %iv to i16
  br i1 %c, label %loop.next, label %loop.latch

loop.next:                                        ; preds = %loop.header
  br label %loop.latch

loop.latch:                                       ; preds = %loop.next, %loop.header
  %blend = phi i16 [ undef, %loop.header ], [ %iv.trunc.2, %loop.next ]
  %dst.ptr = getelementptr inbounds [32 x i16], [32 x i16]* @dst, i16 0, i16 %blend
  store i16 0, i16* %dst.ptr
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp439 = icmp ult i64 %iv, 31
  br i1 %cmp439, label %loop.header, label %exit

exit:                                             ; preds = %loop.latch
  ret void
}

define void @blend_uniform_iv(i1 %c) {
; CHECK-LABEL: @blend_uniform_iv(
; CHECK:       vector.ph:
; CHECK-NEXT:    [[MASK0:%.*]] = insertelement <4 x i1> poison, i1 %c, i32 0
; CHECK-NEXT:    [[MASK1:%.*]] = shufflevector <4 x i1> [[MASK0]], <4 x i1> poison, <4 x i32> zeroinitializer

; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT1:%.*]] = insertelement <4 x i64> poison, i64 [[TMP0]], i32 0
; CHECK-NEXT:    [[BROADCAST_SPLAT2:%.*]] = shufflevector <4 x i64> [[BROADCAST_SPLATINSERT1]], <4 x i64> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    [[TMP1:%.*]] = xor <4 x i1> [[MASK1]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <4 x i1> [[MASK1]], <4 x i64> [[BROADCAST_SPLAT2]], <4 x i64> undef
; CHECK-NEXT:    [[TMP2:%.*]] = extractelement <4 x i64> [[PREDPHI]], i32 0
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds [32 x i16], [32 x i16]* @dst, i16 0, i64 [[TMP2]]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds i16, i16* [[TMP3]], i32 0
; CHECK-NEXT:    [[TMP5:%.*]] = bitcast i16* [[TMP4]] to <4 x i16>*
; CHECK-NEXT:    store <4 x i16> zeroinitializer, <4 x i16>* [[TMP5]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 4
; CHECK-NEXT:    [[TMP6:%.*]] = icmp eq i64 [[INDEX_NEXT]], 32
; CHECK-NEXT:    br i1 [[TMP6]], label %middle.block, label %vector.body
;
entry:
  br label %loop.header

loop.header:                                      ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  br i1 %c, label %loop.next, label %loop.latch

loop.next:                                        ; preds = %loop.header
  br label %loop.latch

loop.latch:                                       ; preds = %loop.next, %loop.header
  %blend = phi i64 [ undef, %loop.header ], [ %iv, %loop.next ]
  %dst.ptr = getelementptr inbounds [32 x i16], [32 x i16]* @dst, i16 0, i64 %blend
  store i16 0, i16* %dst.ptr
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp439 = icmp ult i64 %iv, 31
  br i1 %cmp439, label %loop.header, label %exit

exit:                                             ; preds = %loop.latch
  ret void
}

define void @blend_chain_iv(i1 %c) {
; CHECK-LABEL: @crash3(
; CHECK:       vector.ph:
; CHECK-NEXT:    [[MASK0:%.*]] = insertelement <4 x i1> poison, i1 %c, i32 0
; CHECK-NEXT:    [[MASK1:%.*]] = shufflevector <4 x i1> [[MASK0]], <4 x i1> poison, <4 x i32> zeroinitializer

; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %vector.body ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[INDEX]], 1
; CHECK-NEXT:    [[TMP2:%.*]] = add i64 [[INDEX]], 2
; CHECK-NEXT:    [[TMP3:%.*]] = add i64 [[INDEX]], 3
; CHECK-NEXT:    [[TMP4:%.*]] = xor <4 x i1> [[MASK1]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[TMP5:%.*]] = select <4 x i1> [[MASK1]], <4 x i1> [[TMP4]], <4 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP6:%.*]] = select <4 x i1> [[MASK1]], <4 x i1> [[MASK1]], <4 x i1> zeroinitializer
; CHECK-NEXT:    [[PREDPHI:%.*]] = select <4 x i1> [[TMP6]], <4 x i64> [[VEC_IND]], <4 x i64> undef
; CHECK-NEXT:    [[TMP7:%.*]] = xor <4 x i1> [[MASK1]], <i1 true, i1 true, i1 true, i1 true>
; CHECK-NEXT:    [[TMP8:%.*]] = or <4 x i1> [[TMP6]], [[TMP5]]
; CHECK-NEXT:    [[PREDPHI1:%.*]] = select <4 x i1> [[TMP8]], <4 x i64> [[PREDPHI]], <4 x i64> undef
; CHECK-NEXT:    [[TMP9:%.*]] = extractelement <4 x i64> [[PREDPHI1]], i32 0
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds [32 x i16], [32 x i16]* @dst, i16 0, i64 [[TMP9]]
; CHECK-NEXT:    [[TMP11:%.*]] = extractelement <4 x i64> [[PREDPHI1]], i32 1
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds [32 x i16], [32 x i16]* @dst, i16 0, i64 [[TMP11]]
; CHECK-NEXT:    [[TMP13:%.*]] = extractelement <4 x i64> [[PREDPHI1]], i32 2
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds [32 x i16], [32 x i16]* @dst, i16 0, i64 [[TMP13]]
; CHECK-NEXT:    [[TMP15:%.*]] = extractelement <4 x i64> [[PREDPHI1]], i32 3
; CHECK-NEXT:    [[TMP16:%.*]] = getelementptr inbounds [32 x i16], [32 x i16]* @dst, i16 0, i64 [[TMP15]]
; CHECK-NEXT:    store i16 0, i16* [[TMP10]], align 2
; CHECK-NEXT:    store i16 0, i16* [[TMP12]], align 2
; CHECK-NEXT:    store i16 0, i16* [[TMP14]], align 2
; CHECK-NEXT:    store i16 0, i16* [[TMP16]], align 2
; CHECK-NEXT:    [[INDEX_NEXT]] = add i64 [[INDEX]], 4
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <4 x i64> [[VEC_IND]], <i64 4, i64 4, i64 4, i64 4>
; CHECK-NEXT:    [[TMP17:%.*]] = icmp eq i64 [[INDEX_NEXT]], 32
; CHECK-NEXT:    br i1 [[TMP17]], label %middle.block, label %vector.body
;
entry:
  br label %loop.header

loop.header:                                      ; preds = %loop.latch, %entry
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  br i1 %c, label %loop.next, label %loop.latch

loop.next:                                        ; preds = %loop.header
  br i1 %c, label %loop.next.2, label %loop.next.3

loop.next.2:
  br label %loop.next.3

loop.next.3:
  %blend.1 = phi i64 [ undef, %loop.next ], [ %iv, %loop.next.2 ]
  br label %loop.latch

loop.latch:                                       ; preds = %loop.next, %loop.header
  %blend = phi i64 [ undef, %loop.header ], [ %blend.1, %loop.next.3 ]
  %dst.ptr = getelementptr inbounds [32 x i16], [32 x i16]* @dst, i16 0, i64 %blend
  store i16 0, i16* %dst.ptr
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp439 = icmp ult i64 %iv, 31
  br i1 %cmp439, label %loop.header, label %exit

exit:                                             ; preds = %loop.latch
  ret void
}
