; RUN: llc -mtriple=thumbv8.1m.main-arm-none-eabi -mattr=+mve.fp %s -o - | FileCheck %s

define arm_aapcs_vfpcc <4 x i32> @vmlau32(<4 x i32> %A, <4 x i32> %B, i32 %X) nounwind {
; CHECK-LABEL: vmlau32:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmla.u32 q0, q1, r0
; CHECK-NEXT:    bx lr
entry:
  %0 = insertelement <4 x i32> undef, i32 %X, i32 0
  %1 = shufflevector <4 x i32> %0, <4 x i32> undef, <4 x i32> zeroinitializer
  %2 = mul nsw <4 x i32> %B, %1
  %3 = add nsw <4 x i32> %A, %2
  ret <4 x i32> %3
}

define arm_aapcs_vfpcc <4 x i32> @vmlau32b(<4 x i32> %A, <4 x i32> %B, i32 %X) nounwind {
; CHECK-LABEL: vmlau32b:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmla.u32 q0, q1, r0
; CHECK-NEXT:    bx lr
entry:
  %0 = insertelement <4 x i32> undef, i32 %X, i32 0
  %1 = shufflevector <4 x i32> %0, <4 x i32> undef, <4 x i32> zeroinitializer
  %2 = mul nsw <4 x i32> %1, %B
  %3 = add nsw <4 x i32> %2, %A
  ret <4 x i32> %3
}

define arm_aapcs_vfpcc <8 x i16> @vmlau16(<8 x i16> %A, <8 x i16> %B, i16 %X) nounwind {
; CHECK-LABEL: vmlau16:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmla.u16 q0, q1, r0
; CHECK-NEXT:    bx lr
entry:
  %0 = insertelement <8 x i16> undef, i16 %X, i32 0
  %1 = shufflevector <8 x i16> %0, <8 x i16> undef, <8 x i32> zeroinitializer
  %2 = mul nsw <8 x i16> %B, %1
  %3 = add nsw <8 x i16> %A, %2
  ret <8 x i16> %3
}

define arm_aapcs_vfpcc <8 x i16> @vmlau16b(<8 x i16> %A, <8 x i16> %B, i16 %X) nounwind {
; CHECK-LABEL: vmlau16b:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmla.u16 q0, q1, r0
; CHECK-NEXT:    bx lr
entry:
  %0 = insertelement <8 x i16> undef, i16 %X, i32 0
  %1 = shufflevector <8 x i16> %0, <8 x i16> undef, <8 x i32> zeroinitializer
  %2 = mul nsw <8 x i16> %1, %B
  %3 = add nsw <8 x i16> %2, %A
  ret <8 x i16> %3
}

define arm_aapcs_vfpcc <16 x i8> @vmlau8(<16 x i8> %A, <16 x i8> %B, i8 %X) nounwind {
; CHECK-LABEL: vmlau8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmla.u8 q0, q1, r0
; CHECK-NEXT:    bx lr
entry:
  %0 = insertelement <16 x i8> undef, i8 %X, i32 0
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> zeroinitializer
  %2 = mul nsw <16 x i8> %B, %1
  %3 = add nsw <16 x i8> %A, %2
  ret <16 x i8> %3
}

define arm_aapcs_vfpcc <16 x i8> @vmlau8b(<16 x i8> %A, <16 x i8> %B, i8 %X) nounwind {
; CHECK-LABEL: vmlau8b:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vmla.u8 q0, q1, r0
; CHECK-NEXT:    bx lr
entry:
  %0 = insertelement <16 x i8> undef, i8 %X, i32 0
  %1 = shufflevector <16 x i8> %0, <16 x i8> undef, <16 x i32> zeroinitializer
  %2 = mul nsw <16 x i8> %1, %B
  %3 = add nsw <16 x i8> %2, %A
  ret <16 x i8> %3
}

