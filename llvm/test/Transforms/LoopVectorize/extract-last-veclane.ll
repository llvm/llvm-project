; RUN: opt -loop-vectorize -dce -instcombine -S -force-vector-width=4 < %s 2>%t | FileCheck %s

define void @inv_store_last_lane(i32* noalias nocapture %a, i32* noalias nocapture %inv, i32* noalias nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @inv_store_last_lane
; CHECK: vector.body:
; CHECK:  store <4 x i32> %[[VEC_VAL:.*]], <
; CHECK: middle.block:
; CHECK:  %{{.*}} = extractelement <4 x i32> %[[VEC_VAL]], i32 3

entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %mul = shl nsw i32 %0, 1
  %arrayidx2 = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:              ; preds = %for.body
  %arrayidx5 = getelementptr inbounds i32, i32* %inv, i64 42
  store i32 %mul, i32* %arrayidx5, align 4
  ret void
}

define float @ret_last_lane(float* noalias nocapture %a, float* noalias nocapture readonly %b, i64 %n) {
; CHECK-LABEL: @ret_last_lane
; CHECK: vector.body:
; CHECK:  store <4 x float> %[[VEC_VAL:.*]], <
; CHECK: middle.block:
; CHECK:  %{{.*}} = extractelement <4 x float> %[[VEC_VAL]], i32 3

entry:
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %b, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %mul = fmul float %0, 2.000000e+00
  %arrayidx2 = getelementptr inbounds float, float* %a, i64 %indvars.iv
  store float %mul, float* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                 ; preds = %for.body, %entry
  ret float %mul
}
