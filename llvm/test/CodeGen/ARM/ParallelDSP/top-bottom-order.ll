; RUN: opt -mtriple=thumbv8m.main -mcpu=cortex-m33 -arm-parallel-dsp -S %s -o - | FileCheck %s
; RUN: opt -mtriple=thumbv7a-linux-android -arm-parallel-dsp -S %s -o - | FileCheck %s

; CHECK-LABEL: reorder_gep_arguments 
; CHECK: [[Sub:%[^ ]+]] = xor i32 %iv, -1
; CHECK: [[IdxPtr:%[^ ]+]] = getelementptr inbounds i16, i16* %arrayidx.us, i32 [[Sub]]
; CHECK: [[IdxPtrCast:%[^ ]+]] = bitcast i16* [[IdxPtr]] to i32*
; CHECK: [[Idx:%[^ ]+]] = load i32, i32* [[IdxPtrCast]], align 2
; CHECK: [[Top:%[^ ]+]] = ashr i32 [[Idx]], 16
; CHECK: [[Shl:%[^ ]+]] = shl i32 [[Idx]], 16
; CHECK: [[Bottom:%[^ ]+]] = ashr i32 [[Shl]], 16
; CHECK: [[BPtr:%[^ ]+]] = getelementptr inbounds i16, i16* %B, i32 %iv
; CHECK: [[BData:%[^ ]+]] = load i16, i16* [[BPtr]], align 2
; CHECK: [[BSext:%[^ ]+]] = sext i16 [[BData]] to i32
; CHECK: [[Mul0:%[^ ]+]] = mul nsw i32 [[BSext]], [[Top]]
; CHECK: [[BPtr1:%[^ ]+]] = getelementptr inbounds i16, i16* %B, i32 %add48.us
; CHECK: [[BData1:%[^ ]+]] = load i16, i16* [[BPtr1]], align 2
; CHECK: [[B1Sext:%[^ ]+]] = sext i16 [[BData1]] to i32
; CHECK: [[Mul1:%[^ ]+]] = mul nsw i32 [[B1Sext]], [[Bottom]]

define i32 @reorder_gep_arguments(i16* %B, i16* %arrayidx.us, i32 %d) {
entry:
  br label %for.body36.us

for.body36.us:
  %iv = phi i32 [ %add53.us, %for.body36.us ], [ 5, %entry ]
  %out32_Q12.0114.us = phi i32 [ %add52.us, %for.body36.us ], [ 0, %entry ]
  %sub37.us = sub nsw i32 0, %iv
  %arrayidx38.us = getelementptr inbounds i16, i16* %arrayidx.us, i32 %sub37.us
  %0 = load i16, i16* %arrayidx38.us, align 2
  %conv39.us = sext i16 %0 to i32
  %arrayidx40.us = getelementptr inbounds i16, i16* %B, i32 %iv
  %1 = load i16, i16* %arrayidx40.us, align 2
  %conv41.us = sext i16 %1 to i32
  %mul42.us = mul nsw i32 %conv41.us, %conv39.us
  %add43.us = add i32 %mul42.us, %out32_Q12.0114.us
  %sub45.us = xor i32 %iv, -1
  %arrayidx46.us = getelementptr inbounds i16, i16* %arrayidx.us, i32 %sub45.us
  %2 = load i16, i16* %arrayidx46.us, align 2
  %conv47.us = sext i16 %2 to i32
  %add48.us = or i32 %iv, 1
  %arrayidx49.us = getelementptr inbounds i16, i16* %B, i32 %add48.us
  %3 = load i16, i16* %arrayidx49.us, align 2
  %conv50.us = sext i16 %3 to i32
  %mul51.us = mul nsw i32 %conv50.us, %conv47.us
  %add52.us = add i32 %add43.us, %mul51.us
  %add53.us = add nuw nsw i32 %iv, 2
  %cmp34.us = icmp slt i32 %add53.us, %d
  br i1 %cmp34.us, label %for.body36.us, label %exit

exit:
  ret i32 %add52.us
}

