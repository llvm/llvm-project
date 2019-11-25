; RUN: llc -mtriple=thumbv8.1m.main-arm-none-eabi -mattr=+mve %s -o - | FileCheck %s

define arm_aapcs_vfpcc <16 x i8> @vqneg_test16(<16 x i8> %A) nounwind {
; CHECK-LABEL: vqneg_test16:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vqneg.s8 q0, q0
; CHECK-NEXT:    bx lr
entry:

  %0 = icmp eq <16 x i8> %A, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  %1 = sub nsw <16 x i8> zeroinitializer, %A
  %2 = select <16 x i1> %0, <16 x i8> <i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127>, <16 x i8> %1

  ret <16 x i8> %2
}

define arm_aapcs_vfpcc <8 x i16> @vqneg_test8(<8 x i16> %A) nounwind {
; CHECK-LABEL: vqneg_test8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vqneg.s16 q0, q0
; CHECK-NEXT:    bx lr
entry:

  %0 = icmp eq <8 x i16> %A, <i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  %1 = sub nsw <8 x i16> zeroinitializer, %A
  %2 = select <8 x i1> %0, <8 x i16> <i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767>, <8 x i16> %1

  ret <8 x i16> %2
}

define arm_aapcs_vfpcc <4 x i32> @vqneg_test4(<4 x i32> %A) nounwind {
; CHECK-LABEL: vqneg_test4:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vqneg.s32 q0, q0
; CHECK-NEXT:    bx lr
entry:

  %0 = icmp eq <4 x i32> %A, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %1 = sub nsw <4 x i32> zeroinitializer, %A
  %2 = select <4 x i1> %0, <4 x i32> <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>, <4 x i32> %1

  ret <4 x i32> %2
}

