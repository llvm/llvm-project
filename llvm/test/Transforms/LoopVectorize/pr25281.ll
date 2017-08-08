; RUN: opt < %s  -scev-aa -loop-vectorize -print-alias-sets -S  -o - 2>&1 | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; PR25281
; Just check that we don't crash on this test.
; CHECK-LABEL: @foo
define void @foo(float** noalias nocapture readonly %in, i32* noalias nocapture readonly %isCompressed, float* noalias nocapture readonly %out) {
entry_block:
  %tmp = getelementptr float*, float** %in, i32 0
  %in_0 = load float*, float** %tmp, !alias.scope !0
  %tmp1 = getelementptr i32, i32* %isCompressed, i32 0
  %isCompressed_0 = load i32, i32* %tmp1, !alias.scope !1
  %tmp2 = getelementptr float*, float** %in, i32 1
  %in_1 = load float*, float** %tmp2, !alias.scope !2
  %tmp3 = getelementptr i32, i32* %isCompressed, i32 1
  %isCompressed_1 = load i32, i32* %tmp3, !alias.scope !3
  br label %for_each_frames

for_each_frames:
  %frameIndex = phi i32 [ 0, %entry_block ], [ %nextFrameIndex, %for_each_frames_end ]
  %nextFrameIndex = add nuw nsw i32 %frameIndex, 2
  br label %for_each_channel

for_each_channel:
  %channelIndex = phi i32 [ 0, %for_each_frames ], [ %nextChannelIndex, %for_each_channel ]
  %nextChannelIndex = add nuw nsw i32 %channelIndex, 1
  %tmp4 = add i32 %frameIndex, %channelIndex
  %tmp5 = xor i32 %isCompressed_0, 1
  %tmp6 = mul i32 %frameIndex, %tmp5
  %offset0 = add i32 %tmp6, %channelIndex
  %tmp7 = getelementptr float, float* %in_0, i32 %offset0
  %in_0_index = load float, float* %tmp7, align 4, !alias.scope !4
  %tmp8 = xor i32 %isCompressed_1, 1
  %tmp9 = mul i32 %frameIndex, %tmp8
  %offset1 = add i32 %tmp9, %channelIndex
  %tmp10 = getelementptr float, float* %in_1, i32 %offset1
  %in_1_index = load float, float* %tmp10, align 4, !alias.scope !5
  %tmp11 = fadd float %in_0_index, %in_1_index
  %tmp12 = getelementptr float, float* %out, i32 %tmp4
  store float %tmp11, float* %tmp12, align 4, !alias.noalias !6
  %tmp13 = icmp eq i32 %nextChannelIndex, 2
  br i1 %tmp13, label %for_each_frames_end, label %for_each_channel

for_each_frames_end:
  %tmp14 = icmp eq i32 %nextFrameIndex, 512
  br i1 %tmp14, label %return, label %for_each_frames

return:
  ret void
}

!0 = distinct !{!0}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !0}
!4 = distinct !{!4, !0}
!5 = distinct !{!5, !0}
!6 = !{!2, !3, !4, !5, !1}
