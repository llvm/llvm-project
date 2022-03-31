; RUN: opt -S -loop-vectorize -force-vector-width=4 %s | FileCheck %s

; CHECK-LABEL: @test_fshl
; CHECK-LABEL: vector.body:
; CHECK-NEXT:    [[IDX:%.+]] = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
; CHECK-NEXT:    [[IDX0:%.+]] = add i32 %index, 0
; CHECK-NEXT:    [[FSHL:%.+]] = call <4 x i16> @llvm.fshl.v4i16(<4 x i16> undef, <4 x i16> undef, <4 x i16> <i16 15, i16 15, i16 15, i16 15>)
; CHECK-NEXT:    [[GEP0:%.+]] = getelementptr inbounds i16, i16* %dst, i32 [[IDX0]]
; CHECK-NEXT:    [[GEP1:%.+]] = getelementptr inbounds i16, i16* [[GEP0]], i32 0
; CHECK-NEXT:    [[GEP_BC:%.+]] = bitcast i16* [[GEP1]] to <4 x i16>*
; CHECK-NEXT:    store <4 x i16> [[FSHL]], <4 x i16>* [[GEP_BC]], align 2
; CHECK-NEXT:    [[IDX_NEXT:%.+]] = add nuw i32 [[IDX]], 4
; CHECK-NEXT:    [[EC:%.+]] = icmp eq i32 [[IDX_NEXT]], %n.vec
; CHECK-NEXT:    br i1 [[EC]], label %middle.block, label %vector.body
;
define void @test_fshl(i32 %width, i16* %dst) {
entry:
  br label %for.body9.us.us

for.cond6.for.cond.cleanup8_crit_edge.us.us:      ; preds = %for.body9.us.us
  ret void

for.body9.us.us:                                  ; preds = %for.body9.us.us, %entry
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body9.us.us ]
  %conv4.i.us.us = tail call i16 @llvm.fshl.i16(i16 undef, i16 undef, i16 15)
  %dst.gep = getelementptr inbounds i16, i16* %dst, i32 %iv
  store i16 %conv4.i.us.us, i16* %dst.gep
  %iv.next = add nuw i32 %iv, 1
  %exitcond50 = icmp eq i32 %iv.next, %width
  br i1 %exitcond50, label %for.cond6.for.cond.cleanup8_crit_edge.us.us, label %for.body9.us.us
}

declare i16 @llvm.fshl.i16(i16, i16, i16)
