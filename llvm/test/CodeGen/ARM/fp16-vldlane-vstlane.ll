; RUN: llc -mtriple=arm-eabi -mattr=+armv8.2-a,+fullfp16,+neon -float-abi=hard -O1 < %s | FileCheck %s
; RUN: llc -mtriple=arm-eabi -mattr=+armv8.2-a,+fullfp16,+neon -float-abi=soft -O1 < %s | FileCheck %s

define <4 x half> @vld1d_lane_f16(ptr %pa, <4 x half> %v4) nounwind {
; CHECK-LABEL: vld1d_lane_f16:
; CHECK: vld1.16 {d{{[0-9]+}}[3]}, [r0:16]
entry:
  %a = load half, ptr %pa
  %res = insertelement <4 x half> %v4, half %a, i32 3
  ret <4 x half> %res
}

define <8 x half> @vld1q_lane_f16_1(ptr %pa, <8 x half> %v8) nounwind {
; CHECK-LABEL: vld1q_lane_f16_1:
; CHECK: vld1.16 {d{{[0-9]+}}[1]}, [r0:16]
entry:
  %a = load half, ptr %pa
  %res = insertelement <8 x half> %v8, half %a, i32 1
  ret <8 x half> %res
}

define <8 x half> @vld1q_lane_f16_7(ptr %pa, <8 x half> %v8) nounwind {
; CHECK-LABEL: vld1q_lane_f16_7:
; CHECK: vld1.16 {d{{[0-9]+}}[3]}, [r0:16]
entry:
  %a = load half, ptr %pa
  %res = insertelement <8 x half> %v8, half %a, i32 7
  ret <8 x half> %res
}

define void @vst1d_lane_f16(ptr %pa, <4 x half> %v4) nounwind {
; CHECK-LABEL: vst1d_lane_f16:
; CHECK: vst1.16 {d{{[0-9]+}}[3]}, [r0:16]
entry:
  %a = extractelement <4 x half> %v4, i32 3
  store half %a, ptr %pa
  ret void
}

define void @vst1q_lane_f16_7(ptr %pa, <8 x half> %v8) nounwind {
; CHECK-LABEL: vst1q_lane_f16_7:
; CHECK: vst1.16 {d{{[0-9]+}}[3]}, [r0:16]
entry:
  %a = extractelement <8 x half> %v8, i32 7
  store half %a, ptr %pa
  ret void
}

define void @vst1q_lane_f16_1(ptr %pa, <8 x half> %v8) nounwind {
; CHECK-LABEL: vst1q_lane_f16_1:
; CHECK: vst1.16 {d{{[0-9]+}}[1]}, [r0:16]
entry:
  %a = extractelement <8 x half> %v8, i32 1
  store half %a, ptr %pa
  ret void
}
