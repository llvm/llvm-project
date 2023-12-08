; RUN: llc -mtriple=arm-eabi -mattr=+neon %s -o - | FileCheck %s

define <8 x i8> @vtrni8(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrni8:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vldr d17, [r0]
; CHECK-NEXT:    vtrn.8 d17, d16
; CHECK-NEXT:    vadd.i8 d16, d17, d16
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i8>, ptr %A
  %tmp2 = load <8 x i8>, ptr %B
  %tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  %tmp4 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  %tmp5 = add <8 x i8> %tmp3, %tmp4
  ret <8 x i8> %tmp5
}

define <16 x i8> @vtrni8_Qres(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrni8_Qres:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr [[LDR1:d[0-9]+]], [r1]
; CHECK-NEXT:    vldr [[LDR0:d[0-9]+]], [r0]
; CHECK-NEXT:    vtrn.8 [[LDR0]], [[LDR1]]
; CHECK-NEXT:    vmov r0, r1, [[LDR0]]
; CHECK-NEXT:    vmov r2, r3, [[LDR1]]
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i8>, ptr %A
  %tmp2 = load <8 x i8>, ptr %B
  %tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <16 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14, i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <16 x i8> %tmp3
}

define <4 x i16> @vtrni16(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrni16:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vldr d17, [r0]
; CHECK-NEXT:    vtrn.16 d17, d16
; CHECK-NEXT:    vadd.i16 d16, d17, d16
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x i16>, ptr %A
  %tmp2 = load <4 x i16>, ptr %B
  %tmp3 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %tmp4 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  %tmp5 = add <4 x i16> %tmp3, %tmp4
  ret <4 x i16> %tmp5
}

define <8 x i16> @vtrni16_Qres(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrni16_Qres:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr [[LDR1:d[0-9]+]], [r1]
; CHECK-NEXT:    vldr [[LDR0:d[0-9]+]], [r0]
; CHECK-NEXT:    vtrn.16 [[LDR0]], [[LDR1]]
; CHECK-NEXT:    vmov r0, r1, [[LDR0]]
; CHECK-NEXT:    vmov r2, r3, [[LDR1]]
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x i16>, ptr %A
  %tmp2 = load <4 x i16>, ptr %B
  %tmp3 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <8 x i32> <i32 0, i32 4, i32 2, i32 6, i32 1, i32 5, i32 3, i32 7>
  ret <8 x i16> %tmp3
}

define <2 x i32> @vtrni32(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrni32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vldr d17, [r0]
; CHECK-NEXT:    vtrn.32 d17, d16
; CHECK-NEXT:    vmul.i32 d16, d17, d16
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <2 x i32>, ptr %A
  %tmp2 = load <2 x i32>, ptr %B
  %tmp3 = shufflevector <2 x i32> %tmp1, <2 x i32> %tmp2, <2 x i32> <i32 0, i32 2>
  %tmp4 = shufflevector <2 x i32> %tmp1, <2 x i32> %tmp2, <2 x i32> <i32 1, i32 3>
  %tmp5 = mul <2 x i32> %tmp3, %tmp4
  ret <2 x i32> %tmp5
}

define <4 x i32> @vtrni32_Qres(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrni32_Qres:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr [[LDR1:d[0-9]+]], [r1]
; CHECK-NEXT:    vldr [[LDR0:d[0-9]+]], [r0]
; CHECK-NEXT:    vtrn.32 [[LDR0]], [[LDR1]]
; CHECK-NEXT:    vmov r0, r1, [[LDR0]]
; CHECK-NEXT:    vmov r2, r3, [[LDR1]]
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <2 x i32>, ptr %A
  %tmp2 = load <2 x i32>, ptr %B
  %tmp3 = shufflevector <2 x i32> %tmp1, <2 x i32> %tmp2, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  ret <4 x i32> %tmp3
}

define <2 x float> @vtrnf(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrnf:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vldr d17, [r0]
; CHECK-NEXT:    vtrn.32 d17, d16
; CHECK-NEXT:    vadd.f32 d16, d17, d16
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <2 x float>, ptr %A
  %tmp2 = load <2 x float>, ptr %B
  %tmp3 = shufflevector <2 x float> %tmp1, <2 x float> %tmp2, <2 x i32> <i32 0, i32 2>
  %tmp4 = shufflevector <2 x float> %tmp1, <2 x float> %tmp2, <2 x i32> <i32 1, i32 3>
  %tmp5 = fadd <2 x float> %tmp3, %tmp4
  ret <2 x float> %tmp5
}

define <4 x float> @vtrnf_Qres(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrnf_Qres:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr [[LDR1:d[0-9]+]], [r1]
; CHECK-NEXT:    vldr [[LDR0:d[0-9]+]], [r0]
; CHECK-NEXT:    vtrn.32 [[LDR0]], [[LDR1]]
; CHECK-NEXT:    vmov r0, r1, [[LDR0]]
; CHECK-NEXT:    vmov r2, r3, [[LDR1]]
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <2 x float>, ptr %A
  %tmp2 = load <2 x float>, ptr %B
  %tmp3 = shufflevector <2 x float> %tmp1, <2 x float> %tmp2, <4 x i32> <i32 0, i32 2, i32 1, i32 3>
  ret <4 x float> %tmp3
}

define <16 x i8> @vtrnQi8(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrnQi8:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r0]
; CHECK-NEXT:    vtrn.8 q9, q8
; CHECK-NEXT:    vadd.i8 q8, q9, q8
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <16 x i8>, ptr %A
  %tmp2 = load <16 x i8>, ptr %B
  %tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30>
  %tmp4 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
  %tmp5 = add <16 x i8> %tmp3, %tmp4
  ret <16 x i8> %tmp5
}

define <32 x i8> @vtrnQi8_QQres(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrnQi8_QQres:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vtrn.8 q9, q8
; CHECK-NEXT:    vst1.8 {d18, d19}, [r0:128]!
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <16 x i8>, ptr %A
  %tmp2 = load <16 x i8>, ptr %B
  %tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <32 x i32> <i32 0, i32 16, i32 2, i32 18, i32 4, i32 20, i32 6, i32 22, i32 8, i32 24, i32 10, i32 26, i32 12, i32 28, i32 14, i32 30, i32 1, i32 17, i32 3, i32 19, i32 5, i32 21, i32 7, i32 23, i32 9, i32 25, i32 11, i32 27, i32 13, i32 29, i32 15, i32 31>
  ret <32 x i8> %tmp3
}

define <8 x i16> @vtrnQi16(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrnQi16:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r0]
; CHECK-NEXT:    vtrn.16 q9, q8
; CHECK-NEXT:    vadd.i16 q8, q9, q8
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i16>, ptr %A
  %tmp2 = load <8 x i16>, ptr %B
  %tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14>
  %tmp4 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  %tmp5 = add <8 x i16> %tmp3, %tmp4
  ret <8 x i16> %tmp5
}

define <16 x i16> @vtrnQi16_QQres(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrnQi16_QQres:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vtrn.16 q9, q8
; CHECK-NEXT:    vst1.16 {d18, d19}, [r0:128]!
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i16>, ptr %A
  %tmp2 = load <8 x i16>, ptr %B
  %tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <16 x i32> <i32 0, i32 8, i32 2, i32 10, i32 4, i32 12, i32 6, i32 14, i32 1, i32 9, i32 3, i32 11, i32 5, i32 13, i32 7, i32 15>
  ret <16 x i16> %tmp3
}

define <4 x i32> @vtrnQi32(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrnQi32:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r0]
; CHECK-NEXT:    vtrn.32 q9, q8
; CHECK-NEXT:    vadd.i32 q8, q9, q8
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x i32>, ptr %A
  %tmp2 = load <4 x i32>, ptr %B
  %tmp3 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %tmp4 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  %tmp5 = add <4 x i32> %tmp3, %tmp4
  ret <4 x i32> %tmp5
}

define <8 x i32> @vtrnQi32_QQres(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrnQi32_QQres:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vtrn.32 q9, q8
; CHECK-NEXT:    vst1.32 {d18, d19}, [r0:128]!
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x i32>, ptr %A
  %tmp2 = load <4 x i32>, ptr %B
  %tmp3 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <8 x i32> <i32 0, i32 4, i32 2, i32 6, i32 1, i32 5, i32 3, i32 7>
  ret <8 x i32> %tmp3
}

define <4 x float> @vtrnQf(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrnQf:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r0]
; CHECK-NEXT:    vtrn.32 q9, q8
; CHECK-NEXT:    vadd.f32 q8, q9, q8
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x float>, ptr %A
  %tmp2 = load <4 x float>, ptr %B
  %tmp3 = shufflevector <4 x float> %tmp1, <4 x float> %tmp2, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  %tmp4 = shufflevector <4 x float> %tmp1, <4 x float> %tmp2, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  %tmp5 = fadd <4 x float> %tmp3, %tmp4
  ret <4 x float> %tmp5
}

define <8 x float> @vtrnQf_QQres(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrnQf_QQres:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vtrn.32 q9, q8
; CHECK-NEXT:    vst1.32 {d18, d19}, [r0:128]!
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <4 x float>, ptr %A
  %tmp2 = load <4 x float>, ptr %B
  %tmp3 = shufflevector <4 x float> %tmp1, <4 x float> %tmp2, <8 x i32> <i32 0, i32 4, i32 2, i32 6, i32 1, i32 5, i32 3, i32 7>
  ret <8 x float> %tmp3
}


define <8 x i8> @vtrni8_undef(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrni8_undef:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr d16, [r1]
; CHECK-NEXT:    vldr d17, [r0]
; CHECK-NEXT:    vtrn.8 d17, d16
; CHECK-NEXT:    vadd.i8 d16, d17, d16
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i8>, ptr %A
  %tmp2 = load <8 x i8>, ptr %B
  %tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 0, i32 undef, i32 2, i32 10, i32 undef, i32 12, i32 6, i32 14>
  %tmp4 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 1, i32 9, i32 3, i32 11, i32 5, i32 undef, i32 undef, i32 15>
  %tmp5 = add <8 x i8> %tmp3, %tmp4
  ret <8 x i8> %tmp5
}

define <16 x i8> @vtrni8_undef_Qres(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrni8_undef_Qres:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vldr [[LDR1:d[0-9]+]], [r1]
; CHECK-NEXT:    vldr [[LDR0:d[0-9]+]], [r0]
; CHECK-NEXT:    vtrn.8 [[LDR0]], [[LDR1]]
; CHECK-NEXT:    vmov r0, r1, [[LDR0]]
; CHECK-NEXT:    vmov r2, r3, [[LDR1]]
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i8>, ptr %A
  %tmp2 = load <8 x i8>, ptr %B
  %tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <16 x i32> <i32 0, i32 undef, i32 2, i32 10, i32 undef, i32 12, i32 6, i32 14, i32 1, i32 9, i32 3, i32 11, i32 5, i32 undef, i32 undef, i32 15>
  ret <16 x i8> %tmp3
}

define <8 x i16> @vtrnQi16_undef(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrnQi16_undef:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r1]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r0]
; CHECK-NEXT:    vtrn.16 q9, q8
; CHECK-NEXT:    vadd.i16 q8, q9, q8
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i16>, ptr %A
  %tmp2 = load <8 x i16>, ptr %B
  %tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 0, i32 8, i32 undef, i32 undef, i32 4, i32 12, i32 6, i32 14>
  %tmp4 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 1, i32 undef, i32 3, i32 11, i32 5, i32 13, i32 undef, i32 undef>
  %tmp5 = add <8 x i16> %tmp3, %tmp4
  ret <8 x i16> %tmp5
}

define <16 x i16> @vtrnQi16_undef_QQres(ptr %A, ptr %B) nounwind {
; CHECK-LABEL: vtrnQi16_undef_QQres:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vld1.64 {d16, d17}, [r2]
; CHECK-NEXT:    vld1.64 {d18, d19}, [r1]
; CHECK-NEXT:    vtrn.16 q9, q8
; CHECK-NEXT:    vst1.16 {d18, d19}, [r0:128]!
; CHECK-NEXT:    vst1.64 {d16, d17}, [r0:128]
; CHECK-NEXT:    mov pc, lr
  %tmp1 = load <8 x i16>, ptr %A
  %tmp2 = load <8 x i16>, ptr %B
  %tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <16 x i32> <i32 0, i32 8, i32 undef, i32 undef, i32 4, i32 12, i32 6, i32 14, i32 1, i32 undef, i32 3, i32 11, i32 5, i32 13, i32 undef, i32 undef>
  ret <16 x i16> %tmp3
}

define <8 x i16> @vtrn_lower_shufflemask_undef(ptr %A, ptr %B) {
entry:
  ; CHECK-LABEL: vtrn_lower_shufflemask_undef
  ; CHECK: vtrn
	%tmp1 = load <4 x i16>, ptr %A
	%tmp2 = load <4 x i16>, ptr %B
  %0 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 1, i32 5, i32 3, i32 7>
  ret <8 x i16> %0
}

; Here we get a build_vector node, where all the incoming extract_element
; values do modify the type. However, we get different input types, as some of
; them get truncated from i32 to i8 (from comparing cmp0 with cmp1) and some of
; them get truncated from i16 to i8 (from comparing cmp2 with cmp3).
define <8 x i8> @vtrn_mismatched_builvector0(<8 x i8> %tr0, <8 x i8> %tr1,
                                             <4 x i32> %cmp0, <4 x i32> %cmp1,
                                             <4 x i16> %cmp2, <4 x i16> %cmp3) {
  ; CHECK-LABEL: vtrn_mismatched_builvector0:
  ; CHECK: vmovn.i32
  ; CHECK: vbsl
  %c0 = icmp ult <4 x i32> %cmp0, %cmp1
  %c1 = icmp ult <4 x i16> %cmp2, %cmp3
  %c = shufflevector <4 x i1> %c0, <4 x i1> %c1, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %rv = select <8 x i1> %c, <8 x i8> %tr0, <8 x i8> %tr1
  ret <8 x i8> %rv
}

; Here we get a build_vector node, where half the incoming extract_element
; values do not modify the type (the values form cmp2), but half of them do
; (from the icmp operation).
define <8 x i8> @vtrn_mismatched_builvector1(<8 x i8> %tr0, <8 x i8> %tr1,
                           <4 x i32> %cmp0, <4 x i32> %cmp1, ptr %cmp2_ptr) {
  ; CHECK-LABEL: vtrn_mismatched_builvector1:
  ; We need to extend the 4 x i8 to 4 x i16 in order to perform the vtrn
  ; CHECK: vmovl
  ; CHECK: vbsl
  %cmp2_load = load <4 x i8>, ptr %cmp2_ptr, align 4
  %cmp2 = trunc <4 x i8> %cmp2_load to <4 x i1>
  %c0 = icmp ult <4 x i32> %cmp0, %cmp1
  %c = shufflevector <4 x i1> %c0, <4 x i1> %cmp2, <8 x i32> <i32 0, i32 4, i32 1, i32 5, i32 2, i32 6, i32 3, i32 7>
  %rv = select <8 x i1> %c, <8 x i8> %tr0, <8 x i8> %tr1
  ret <8 x i8> %rv
}

; The shuffle mask is half a vtrn; we duplicate the half to produce the
; full result.
define void @lower_twice_no_vtrn(ptr %A, ptr %B, ptr %C) {
entry:
  ; CHECK-LABEL: lower_twice_no_vtrn:
  ; CHECK: @ %bb.0:
  ; CHECK-NEXT: vldr d16, [r1]
  ; CHECK-NEXT: vldr d18, [r0]
  ; CHECK-NEXT: vtrn.16 d18, d16
  ; CHECK-NEXT: vorr d17, d16, d16
  ; CHECK-NEXT: vst1.64 {d16, d17}, [r2]
  ; CHECK-NEXT: mov pc, lr
  %tmp1 = load <4 x i16>, ptr %A
  %tmp2 = load <4 x i16>, ptr %B
  %0 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <8 x i32> <i32 undef, i32 5, i32 3, i32 7, i32 1, i32 5, i32 3, i32 7>
  store <8 x i16> %0, ptr %C
  ret void
}

; The shuffle mask is half a vtrn; we duplicate the half to produce the
; full result.
define void @upper_twice_no_vtrn(ptr %A, ptr %B, ptr %C) {
entry:
  ; CHECK-LABEL: upper_twice_no_vtrn:
  ; CHECK: @ %bb.0:
  ; CHECK-NEXT: vldr d16, [r1]
  ; CHECK-NEXT: vldr d18, [r0]
  ; CHECK-NEXT: vtrn.16 d18, d16
  ; CHECK-NEXT: vorr d19, d18, d18
  ; CHECK-NEXT: vst1.64 {d18, d19}, [r2]
  ; CHECK-NEXT: mov pc, lr
  %tmp1 = load <4 x i16>, ptr %A
  %tmp2 = load <4 x i16>, ptr %B
  %0 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <8 x i32> <i32 0, i32 undef, i32 2, i32 6, i32 0, i32 4, i32 2, i32 6>
  store <8 x i16> %0, ptr %C
  ret void
}
