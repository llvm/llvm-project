; RUN: opt -mtriple=arm-arm-eabi -mcpu=cortex-m33 < %s -arm-parallel-dsp -S | FileCheck %s

; CHECK-LABEL: topbottom_mul
define void @topbottom_mul(i32 %N, i32* noalias nocapture readnone %Out, i16* nocapture readonly %In1, i16* nocapture readonly %In2) {
entry:
  br label %for.body

; CHECK: for.body:
; CHECK: [[Cast_PIn1_0:%[^ ]+]] = bitcast i16* %PIn1.0 to i32*
; CHECK: [[PIn1_01:%[^ ]+]] = load i32, i32* [[Cast_PIn1_0]], align 2
; CHECK: [[PIn1_01_shl:%[^ ]+]] = shl i32 [[PIn1_01]], 16
; CHECK: [[PIn1_0:%[^ ]+]] = ashr i32 [[PIn1_01_shl]], 16
; CHECK: [[PIn1_1:%[^ ]+]] = ashr i32 [[PIn1_01]], 16

; CHECK: [[Cast_PIn2_0:%[^ ]+]] = bitcast i16* %PIn2.0 to i32*
; CHECK: [[PIn2_01:%[^ ]+]] = load i32, i32* [[Cast_PIn2_0]], align 2
; CHECK: [[PIn2_01_shl:%[^ ]+]] = shl i32 [[PIn2_01]], 16
; CHECK: [[PIn2_0:%[^ ]+]] = ashr i32 [[PIn2_01_shl]], 16
; CHECK: [[PIn2_1:%[^ ]+]] = ashr i32 [[PIn2_01]], 16

; CHECK: mul nsw i32 [[PIn1_0]], [[PIn2_0]]
; CHECK: mul nsw i32 [[PIn1_1]], [[PIn2_1]]

; CHECK: [[Cast_PIn1_2:%[^ ]+]] = bitcast i16* %PIn1.2 to i32*
; CHECK: [[PIn1_23:%[^ ]+]] = load i32, i32* [[Cast_PIn1_2]], align 2
; CHECK: [[PIn1_23_shl:%[^ ]+]] = shl i32 [[PIn1_23]], 16
; CHECK: [[PIn1_2:%[^ ]+]] = ashr i32 [[PIn1_23_shl]], 16
; CHECK: [[PIn1_3:%[^ ]+]] = ashr i32 [[PIn1_23]], 16

; CHECK: [[Cast_PIn2_2:%[^ ]+]] = bitcast i16* %PIn2.2 to i32*
; CHECK: [[PIn2_23:%[^ ]+]] = load i32, i32* [[Cast_PIn2_2]], align 2
; CHECK: [[PIn2_23_shl:%[^ ]+]] = shl i32 [[PIn2_23]], 16
; CHECK: [[PIn2_2:%[^ ]+]] = ashr i32 [[PIn2_23_shl]], 16
; CHECK: [[PIn2_3:%[^ ]+]] = ashr i32 [[PIn2_23]], 16

; CHECK: mul nsw i32 [[PIn1_2]], [[PIn2_2]]
; CHECK: mul nsw i32 [[PIn1_3]], [[PIn2_3]]

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %count = phi i32 [ %N, %entry ], [ %count.next, %for.body ]
  %PIn1.0 = getelementptr inbounds i16, i16* %In1, i32 %iv
  %In1.0 = load i16, i16* %PIn1.0, align 2
  %SIn1.0 = sext i16 %In1.0 to i32
  %PIn2.0 = getelementptr inbounds i16, i16* %In2, i32 %iv
  %In2.0 = load i16, i16* %PIn2.0, align 2
  %SIn2.0 = sext i16 %In2.0 to i32
  %mul5.us.i.i = mul nsw i32 %SIn1.0, %SIn2.0
  %Out.0 = getelementptr inbounds i32, i32* %Out, i32 %iv
  store i32 %mul5.us.i.i, i32* %Out.0, align 4
  %iv.1 = or i32 %iv, 1
  %PIn1.1 = getelementptr inbounds i16, i16* %In1, i32 %iv.1
  %In1.1 = load i16, i16* %PIn1.1, align 2
  %SIn1.1 = sext i16 %In1.1 to i32
  %PIn2.1 = getelementptr inbounds i16, i16* %In2, i32 %iv.1
  %In2.1 = load i16, i16* %PIn2.1, align 2
  %SIn2.1 = sext i16 %In2.1 to i32
  %mul5.us.i.1.i = mul nsw i32 %SIn1.1, %SIn2.1
  %Out.1 = getelementptr inbounds i32, i32* %Out, i32 %iv.1
  store i32 %mul5.us.i.1.i, i32* %Out.1, align 4
  %iv.2 = or i32 %iv, 2
  %PIn1.2 = getelementptr inbounds i16, i16* %In1, i32 %iv.2
  %In1.2 = load i16, i16* %PIn1.2, align 2
  %SIn1.2 = sext i16 %In1.2 to i32
  %PIn2.2 = getelementptr inbounds i16, i16* %In2, i32 %iv.2
  %In2.2 = load i16, i16* %PIn2.2, align 2
  %SIn2.2 = sext i16 %In2.2 to i32
  %mul5.us.i.2.i = mul nsw i32 %SIn1.2, %SIn2.2
  %Out.2 = getelementptr inbounds i32, i32* %Out, i32 %iv.2
  store i32 %mul5.us.i.2.i, i32* %Out.2, align 4
  %iv.3 = or i32 %iv, 3
  %PIn1.3 = getelementptr inbounds i16, i16* %In1, i32 %iv.3
  %In1.3 = load i16, i16* %PIn1.3, align 2
  %SIn1.3 = sext i16 %In1.3 to i32
  %PIn2.3 = getelementptr inbounds i16, i16* %In2, i32 %iv.3
  %In2.3 = load i16, i16* %PIn2.3, align 2
  %SIn2.3 = sext i16 %In2.3 to i32
  %mul5.us.i.3.i = mul nsw i32 %SIn1.3, %SIn2.3
  %Out.3 = getelementptr inbounds i32, i32* %Out, i32 %iv.3
  store i32 %mul5.us.i.3.i, i32* %Out.3, align 4
  %iv.next = add i32 %iv, 4
  %count.next = add i32 %count, -4
  %niter375.ncmp.3.i = icmp eq i32 %count.next, 0
  br i1 %niter375.ncmp.3.i, label %exit, label %for.body

exit:
  ret void
}

; CHECK-LABEL: topbottom_mul_load_const
define void @topbottom_mul_load_const(i32 %N, i32* noalias nocapture readnone %Out, i16* nocapture readonly %In, i16* %C) {
entry:
  %const = load i16, i16* %C
  %conv4.i.i = sext i16 %const to i32
  br label %for.body

; CHECK: for.body:
; CHECK: [[Cast_PIn_0:%[^ ]+]] = bitcast i16* %PIn.0 to i32*
; CHECK: [[PIn_01:%[^ ]+]] = load i32, i32* [[Cast_PIn_0]], align 2
; CHECK: [[PIn_01_shl:%[^ ]+]] = shl i32 [[PIn_01]], 16
; CHECK: [[PIn_0:%[^ ]+]] = ashr i32 [[PIn_01_shl]], 16
; CHECK: [[PIn_1:%[^ ]+]] = ashr i32 [[PIn_01]], 16

; CHECK: mul nsw i32 [[PIn_0]], %conv4.i.i
; CHECK: mul nsw i32 [[PIn_1]], %conv4.i.i

; CHECK: [[Cast_PIn_2:%[^ ]+]] = bitcast i16* %PIn.2 to i32*
; CHECK: [[PIn_23:%[^ ]+]] = load i32, i32* [[Cast_PIn_2]], align 2
; CHECK: [[PIn_23_shl:%[^ ]+]] = shl i32 [[PIn_23]], 16
; CHECK: [[PIn_2:%[^ ]+]] = ashr i32 [[PIn_23_shl]], 16
; CHECK: [[PIn_3:%[^ ]+]] = ashr i32 [[PIn_23]], 16

; CHECK: mul nsw i32 [[PIn_2]], %conv4.i.i
; CHECK: mul nsw i32 [[PIn_3]], %conv4.i.i

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %count = phi i32 [ %N, %entry ], [ %count.next, %for.body ]
  %PIn.0 = getelementptr inbounds i16, i16* %In, i32 %iv
  %In.0 = load i16, i16* %PIn.0, align 2
  %conv.us.i144.i = sext i16 %In.0 to i32
  %mul5.us.i.i = mul nsw i32 %conv.us.i144.i, %conv4.i.i
  %Out.0 = getelementptr inbounds i32, i32* %Out, i32 %iv
  store i32 %mul5.us.i.i, i32* %Out.0, align 4
  %iv.1 = or i32 %iv, 1
  %PIn.1 = getelementptr inbounds i16, i16* %In, i32 %iv.1
  %In.1 = load i16, i16* %PIn.1, align 2
  %conv.us.i144.1.i = sext i16 %In.1 to i32
  %mul5.us.i.1.i = mul nsw i32 %conv.us.i144.1.i, %conv4.i.i
  %Out.1 = getelementptr inbounds i32, i32* %Out, i32 %iv.1
  store i32 %mul5.us.i.1.i, i32* %Out.1, align 4
  %iv.2 = or i32 %iv, 2
  %PIn.2 = getelementptr inbounds i16, i16* %In, i32 %iv.2
  %In.3 = load i16, i16* %PIn.2, align 2
  %conv.us.i144.2.i = sext i16 %In.3 to i32
  %mul5.us.i.2.i = mul nsw i32 %conv.us.i144.2.i, %conv4.i.i
  %Out.2 = getelementptr inbounds i32, i32* %Out, i32 %iv.2
  store i32 %mul5.us.i.2.i, i32* %Out.2, align 4
  %iv.3 = or i32 %iv, 3
  %PIn.3 = getelementptr inbounds i16, i16* %In, i32 %iv.3
  %In.4 = load i16, i16* %PIn.3, align 2
  %conv.us.i144.3.i = sext i16 %In.4 to i32
  %mul5.us.i.3.i = mul nsw i32 %conv.us.i144.3.i, %conv4.i.i
  %Out.3 = getelementptr inbounds i32, i32* %Out, i32 %iv.3
  store i32 %mul5.us.i.3.i, i32* %Out.3, align 4
  %iv.next = add i32 %iv, 4
  %count.next = add i32 %count, -4
  %niter375.ncmp.3.i = icmp eq i32 %count.next, 0
  br i1 %niter375.ncmp.3.i, label %exit, label %for.body

exit:
  ret void
}

; CHECK-LABEL: topbottom_mul_64
define void @topbottom_mul_64(i32 %N, i64* noalias nocapture readnone %Out, i16* nocapture readonly %In1, i16* nocapture readonly %In2) {
entry:
  br label %for.body

; CHECK: for.body:
; CHECK: [[Cast_PIn1_0:%[^ ]+]] = bitcast i16* %PIn1.0 to i32*
; CHECK: [[PIn1_01:%[^ ]+]] = load i32, i32* [[Cast_PIn1_0]], align 2
; CHECK: [[PIn1_01_shl:%[^ ]+]] = shl i32 [[PIn1_01]], 16
; CHECK: [[PIn1_0:%[^ ]+]] = ashr i32 [[PIn1_01_shl]], 16
; CHECK: [[PIn1_1:%[^ ]+]] = ashr i32 [[PIn1_01]], 16

; CHECK: [[Cast_PIn2_0:%[^ ]+]] = bitcast i16* %PIn2.0 to i32*
; CHECK: [[PIn2_01:%[^ ]+]] = load i32, i32* [[Cast_PIn2_0]], align 2
; CHECK: [[PIn2_01_shl:%[^ ]+]] = shl i32 [[PIn2_01]], 16
; CHECK: [[PIn2_0:%[^ ]+]] = ashr i32 [[PIn2_01_shl]], 16
; CHECK: [[PIn2_1:%[^ ]+]] = ashr i32 [[PIn2_01]], 16

; CHECK: [[Mul0:%[^ ]+]] = mul nsw i32 [[PIn1_0]], [[PIn2_0]]
; CHECK: [[SMul0:%[^ ]+]] = sext i32 [[Mul0]] to i64
; CHECK: [[Mul1:%[^ ]+]] = mul nsw i32 [[PIn1_1]], [[PIn2_1]]
; CHECK: [[SMul1:%[^ ]+]] = sext i32 [[Mul1]] to i64
; CHECK: add i64 [[SMul0]], [[SMul1]]

; CHECK: [[Cast_PIn1_2:%[^ ]+]] = bitcast i16* %PIn1.2 to i32*
; CHECK: [[PIn1_23:%[^ ]+]] = load i32, i32* [[Cast_PIn1_2]], align 2
; CHECK: [[PIn1_23_shl:%[^ ]+]] = shl i32 [[PIn1_23]], 16
; CHECK: [[PIn1_2:%[^ ]+]] = ashr i32 [[PIn1_23_shl]], 16
; CHECK: [[PIn1_3:%[^ ]+]] = ashr i32 [[PIn1_23]], 16

; CHECK: [[Cast_PIn2_2:%[^ ]+]] = bitcast i16* %PIn2.2 to i32*
; CHECK: [[PIn2_23:%[^ ]+]] = load i32, i32* [[Cast_PIn2_2]], align 2
; CHECK: [[PIn2_23_shl:%[^ ]+]] = shl i32 [[PIn2_23]], 16
; CHECK: [[PIn2_2:%[^ ]+]] = ashr i32 [[PIn2_23_shl]], 16
; CHECK: [[PIn2_3:%[^ ]+]] = ashr i32 [[PIn2_23]], 16

; CHECK: [[Mul2:%[^ ]+]] = mul nsw i32 [[PIn1_2]], [[PIn2_2]]
; CHECK: [[SMul2:%[^ ]+]] = sext i32 [[Mul2]] to i64
; CHECK: [[Mul3:%[^ ]+]] = mul nsw i32 [[PIn1_3]], [[PIn2_3]]
; CHECK: [[SMul3:%[^ ]+]] = sext i32 [[Mul3]] to i64
; CHECK: add i64 [[SMul2]], [[SMul3]]

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %iv.out = phi i32 [ 0, %entry] , [ %iv.out.next, %for.body ]
  %count = phi i32 [ %N, %entry ], [ %count.next, %for.body ]
  %PIn1.0 = getelementptr inbounds i16, i16* %In1, i32 %iv
  %In1.0 = load i16, i16* %PIn1.0, align 2
  %SIn1.0 = sext i16 %In1.0 to i32
  %PIn2.0 = getelementptr inbounds i16, i16* %In2, i32 %iv
  %In2.0 = load i16, i16* %PIn2.0, align 2
  %SIn2.0 = sext i16 %In2.0 to i32
  %mul5.us.i.i = mul nsw i32 %SIn1.0, %SIn2.0
  %sext.0 = sext i32 %mul5.us.i.i to i64
  %iv.1 = or i32 %iv, 1
  %PIn1.1 = getelementptr inbounds i16, i16* %In1, i32 %iv.1
  %In1.1 = load i16, i16* %PIn1.1, align 2
  %SIn1.1 = sext i16 %In1.1 to i32
  %PIn2.1 = getelementptr inbounds i16, i16* %In2, i32 %iv.1
  %In2.1 = load i16, i16* %PIn2.1, align 2
  %SIn2.1 = sext i16 %In2.1 to i32
  %mul5.us.i.1.i = mul nsw i32 %SIn1.1, %SIn2.1
  %sext.1 = sext i32 %mul5.us.i.1.i to i64
  %mac.0 = add i64 %sext.0, %sext.1
  %Out.0 = getelementptr inbounds i64, i64* %Out, i32 %iv.out
  store i64 %mac.0, i64* %Out.0, align 4
  %iv.2 = or i32 %iv, 2
  %PIn1.2 = getelementptr inbounds i16, i16* %In1, i32 %iv.2
  %In1.2 = load i16, i16* %PIn1.2, align 2
  %SIn1.2 = sext i16 %In1.2 to i32
  %PIn2.2 = getelementptr inbounds i16, i16* %In2, i32 %iv.2
  %In2.2 = load i16, i16* %PIn2.2, align 2
  %SIn2.2 = sext i16 %In2.2 to i32
  %mul5.us.i.2.i = mul nsw i32 %SIn1.2, %SIn2.2
  %sext.2 = sext i32 %mul5.us.i.2.i to i64
  %iv.3 = or i32 %iv, 3
  %PIn1.3 = getelementptr inbounds i16, i16* %In1, i32 %iv.3
  %In1.3 = load i16, i16* %PIn1.3, align 2
  %SIn1.3 = sext i16 %In1.3 to i32
  %PIn2.3 = getelementptr inbounds i16, i16* %In2, i32 %iv.3
  %In2.3 = load i16, i16* %PIn2.3, align 2
  %SIn2.3 = sext i16 %In2.3 to i32
  %mul5.us.i.3.i = mul nsw i32 %SIn1.3, %SIn2.3
  %sext.3 = sext i32 %mul5.us.i.3.i to i64
  %mac.1 = add i64 %sext.2, %sext.3
  %iv.out.1 = or i32 %iv.out, 1
  %Out.1 = getelementptr inbounds i64, i64* %Out, i32 %iv.out.1
  store i64 %mac.1, i64* %Out.1, align 4
  %iv.next = add i32 %iv, 4
  %iv.out.next = add i32 %iv.out, 2
  %count.next = add i32 %count, -4
  %niter375.ncmp.3.i = icmp eq i32 %count.next, 0
  br i1 %niter375.ncmp.3.i, label %exit, label %for.body

exit:
  ret void
}
