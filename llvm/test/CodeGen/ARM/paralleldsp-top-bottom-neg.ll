; RUN: opt -mtriple=arm-arm-eabi -mcpu=cortex-m33 < %s -arm-parallel-dsp -S | FileCheck %s

; CHECK-LABEL: topbottom_mul_alias
; CHECK-NOT: bitcast i16*
define void @topbottom_mul_alias(i32 %N, i32* nocapture readnone %Out, i16* nocapture readonly %In1, i16* nocapture readonly %In2) {
entry:
  br label %for.body

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

; TODO: We should be able to handle this by splatting the const value.
; CHECK-LABEL: topbottom_mul_const
; CHECK-NOT: bitcast i16*
define void @topbottom_mul_const(i32 %N, i32* noalias nocapture readnone %Out, i16* nocapture readonly %In, i16 signext %const) {
entry:
  %conv4.i.i = sext i16 %const to i32
  br label %for.body

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

; TODO: We should be able to handle this and use smulwt and smulwb.
; CHECK-LABEL: topbottom_mul_word_load_const
; CHECK-NOT: bitcast i16*
define void @topbottom_mul_word_load_const(i32 %N, i32* noalias nocapture readnone %Out, i16* nocapture readonly %In, i32* %C) {
entry:
  %const = load i32, i32* %C
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %count = phi i32 [ %N, %entry ], [ %count.next, %for.body ]
  %PIn.0 = getelementptr inbounds i16, i16* %In, i32 %iv
  %In.0 = load i16, i16* %PIn.0, align 2
  %conv.us.i144.i = sext i16 %In.0 to i32
  %mul5.us.i.i = mul nsw i32 %conv.us.i144.i, %const
  %Out.0 = getelementptr inbounds i32, i32* %Out, i32 %iv
  store i32 %mul5.us.i.i, i32* %Out.0, align 4
  %iv.1 = or i32 %iv, 1
  %PIn.1 = getelementptr inbounds i16, i16* %In, i32 %iv.1
  %In.1 = load i16, i16* %PIn.1, align 2
  %conv.us.i144.1.i = sext i16 %In.1 to i32
  %mul5.us.i.1.i = mul nsw i32 %conv.us.i144.1.i, %const
  %Out.1 = getelementptr inbounds i32, i32* %Out, i32 %iv.1
  store i32 %mul5.us.i.1.i, i32* %Out.1, align 4
  %iv.2 = or i32 %iv, 2
  %PIn.2 = getelementptr inbounds i16, i16* %In, i32 %iv.2
  %In.3 = load i16, i16* %PIn.2, align 2
  %conv.us.i144.2.i = sext i16 %In.3 to i32
  %mul5.us.i.2.i = mul nsw i32 %conv.us.i144.2.i, %const
  %Out.2 = getelementptr inbounds i32, i32* %Out, i32 %iv.2
  store i32 %mul5.us.i.2.i, i32* %Out.2, align 4
  %iv.3 = or i32 %iv, 3
  %PIn.3 = getelementptr inbounds i16, i16* %In, i32 %iv.3
  %In.4 = load i16, i16* %PIn.3, align 2
  %conv.us.i144.3.i = sext i16 %In.4 to i32
  %mul5.us.i.3.i = mul nsw i32 %conv.us.i144.3.i, %const
  %Out.3 = getelementptr inbounds i32, i32* %Out, i32 %iv.3
  store i32 %mul5.us.i.3.i, i32* %Out.3, align 4
  %iv.next = add i32 %iv, 4
  %count.next = add i32 %count, -4
  %niter375.ncmp.3.i = icmp eq i32 %count.next, 0
  br i1 %niter375.ncmp.3.i, label %exit, label %for.body

exit:
  ret void
}

; CHECK-LABEL: topbottom_mul_8
; CHECK-NOT: bitcast i16*
define void @topbottom_mul_8(i32 %N, i32* noalias nocapture readnone %Out, i8* nocapture readonly %In1, i8* nocapture readonly %In2) {
entry:
  br label %for.body

for.body:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %for.body ]
  %count = phi i32 [ %N, %entry ], [ %count.next, %for.body ]
  %PIn1.0 = getelementptr inbounds i8, i8* %In1, i32 %iv
  %In1.0 = load i8, i8* %PIn1.0, align 1
  %SIn1.0 = sext i8 %In1.0 to i32
  %PIn2.0 = getelementptr inbounds i8, i8* %In2, i32 %iv
  %In2.0 = load i8, i8* %PIn2.0, align 1
  %SIn2.0 = sext i8 %In2.0 to i32
  %mul5.us.i.i = mul nsw i32 %SIn1.0, %SIn2.0
  %Out.0 = getelementptr inbounds i32, i32* %Out, i32 %iv
  store i32 %mul5.us.i.i, i32* %Out.0, align 4
  %iv.1 = or i32 %iv, 1
  %PIn1.1 = getelementptr inbounds i8, i8* %In1, i32 %iv.1
  %In1.1 = load i8, i8* %PIn1.1, align 1
  %SIn1.1 = sext i8 %In1.1 to i32
  %PIn2.1 = getelementptr inbounds i8, i8* %In2, i32 %iv.1
  %In2.1 = load i8, i8* %PIn2.1, align 1
  %SIn2.1 = sext i8 %In2.1 to i32
  %mul5.us.i.1.i = mul nsw i32 %SIn1.1, %SIn2.1
  %Out.1 = getelementptr inbounds i32, i32* %Out, i32 %iv.1
  store i32 %mul5.us.i.1.i, i32* %Out.1, align 4
  %iv.2 = or i32 %iv, 2
  %PIn1.2 = getelementptr inbounds i8, i8* %In1, i32 %iv.2
  %In1.2 = load i8, i8* %PIn1.2, align 1
  %SIn1.2 = sext i8 %In1.2 to i32
  %PIn2.2 = getelementptr inbounds i8, i8* %In2, i32 %iv.2
  %In2.2 = load i8, i8* %PIn2.2, align 1
  %SIn2.2 = sext i8 %In2.2 to i32
  %mul5.us.i.2.i = mul nsw i32 %SIn1.2, %SIn2.2
  %Out.2 = getelementptr inbounds i32, i32* %Out, i32 %iv.2
  store i32 %mul5.us.i.2.i, i32* %Out.2, align 4
  %iv.3 = or i32 %iv, 3
  %PIn1.3 = getelementptr inbounds i8, i8* %In1, i32 %iv.3
  %In1.3 = load i8, i8* %PIn1.3, align 1
  %SIn1.3 = sext i8 %In1.3 to i32
  %PIn2.3 = getelementptr inbounds i8, i8* %In2, i32 %iv.3
  %In2.3 = load i8, i8* %PIn2.3, align 1
  %SIn2.3 = sext i8 %In2.3 to i32
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
