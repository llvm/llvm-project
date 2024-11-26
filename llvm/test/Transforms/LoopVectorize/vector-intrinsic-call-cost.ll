; RUN: opt -S -passes=loop-vectorize -force-vector-width=4 %s | FileCheck %s

; CHECK-LABEL: @test_fshl
; CHECK-LABEL: vector.body:
; CHECK-NEXT:    [[IDX:%.+]] = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK-NEXT:    [[IDX0:%.+]] = add i32 %index, 0
; CHECK-NEXT:    [[GEP:%.+]] = getelementptr inbounds i16, ptr %src, i32 [[IDX0]]
; CHECK-NEXT:    [[GEP0:%.+]] = getelementptr inbounds i16, ptr [[GEP]], i32 0
; CHECK-NEXT:    [[WIDE_LOAD:%.+]] = load <4 x i16>, ptr [[GEP0]], align 2
; CHECK-NEXT:    [[FSHL:%.+]] = call <4 x i16> @llvm.fshl.v4i16(<4 x i16> [[WIDE_LOAD]], <4 x i16> [[WIDE_LOAD]], <4 x i16> splat (i16 15))
; CHECK-NEXT:    [[GEP0:%.+]] = getelementptr inbounds i16, ptr %dst, i32 [[IDX0]]
; CHECK-NEXT:    [[GEP1:%.+]] = getelementptr inbounds i16, ptr [[GEP0]], i32 0
; CHECK-NEXT:    store <4 x i16> [[FSHL]], ptr [[GEP1]], align 2
; CHECK-NEXT:    [[IDX_NEXT:%.+]] = add nuw i32 [[IDX]], 4
; CHECK-NEXT:    [[EC:%.+]] = icmp eq i32 [[IDX_NEXT]], %n.vec
; CHECK-NEXT:    br i1 [[EC]], label %middle.block, label %vector.body
;
define void @test_fshl(i32 %width, ptr %dst, ptr %src) {
entry:
  br label %for.body9.us.us

for.cond6.for.cond.cleanup8_crit_edge.us.us:      ; preds = %for.body9.us.us
  ret void

for.body9.us.us:                                  ; preds = %for.body9.us.us, %entry
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body9.us.us ]
  %gep = getelementptr inbounds i16, ptr %src, i32 %iv
  %l = load i16, ptr %gep
  %conv4.i.us.us = tail call i16 @llvm.fshl.i16(i16 %l, i16 %l, i16 15)
  %dst.gep = getelementptr inbounds i16, ptr %dst, i32 %iv
  store i16 %conv4.i.us.us, ptr %dst.gep
  %iv.next = add nuw i32 %iv, 1
  %exitcond50 = icmp eq i32 %iv.next, %width
  br i1 %exitcond50, label %for.cond6.for.cond.cleanup8_crit_edge.us.us, label %for.body9.us.us
}

declare i16 @llvm.fshl.i16(i16, i16, i16)
