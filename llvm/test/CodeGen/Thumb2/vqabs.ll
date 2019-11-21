; RUN: llc -mtriple=thumbv8.1m.main-arm-none-eabi -mattr=+mve %s -o - | FileCheck %s

define arm_aapcs_vfpcc <16 x i8> @vqabs_test16(<16 x i8> %A) nounwind {
; CHECK-LABEL: vqabs_test16:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vqabs.s8 q0, q0
; CHECK-NEXT:    bx lr
entry:

  %0 = icmp sgt <16 x i8> %A, zeroinitializer
  %1 = icmp eq <16 x i8> %A, <i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128, i8 -128>
  %2 = sub nsw <16 x i8> zeroinitializer, %A
  %3 = select <16 x i1> %1, <16 x i8> <i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127, i8 127>, <16 x i8> %2
  %4 = select <16 x i1> %0, <16 x i8> %A, <16 x i8> %3
  
  ret <16 x i8> %4
}

define arm_aapcs_vfpcc <8 x i16> @vqabs_test8(<8 x i16> %A) nounwind {
; CHECK-LABEL: vqabs_test8:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vqabs.s16 q0, q0
; CHECK-NEXT:    bx lr
entry:

  %0 = icmp sgt <8 x i16> %A, zeroinitializer
  %1 = icmp eq <8 x i16> %A, <i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768, i16 -32768>
  %2 = sub nsw <8 x i16> zeroinitializer, %A
  %3 = select <8 x i1> %1, <8 x i16> <i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767, i16 32767>, <8 x i16> %2
  %4 = select <8 x i1> %0, <8 x i16> %A, <8 x i16> %3
  
  ret <8 x i16> %4
}

define arm_aapcs_vfpcc <4 x i32> @vqabs_test4(<4 x i32> %A) nounwind {
; CHECK-LABEL: vqabs_test4:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    vqabs.s32 q0, q0
; CHECK-NEXT:    bx lr
entry:

  %0 = icmp sgt <4 x i32> %A, zeroinitializer
  %1 = icmp eq <4 x i32> %A, <i32 -2147483648, i32 -2147483648, i32 -2147483648, i32 -2147483648>
  %2 = sub nsw <4 x i32> zeroinitializer, %A
  %3 = select <4 x i1> %1, <4 x i32> <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>, <4 x i32> %2
  %4 = select <4 x i1> %0, <4 x i32> %A, <4 x i32> %3
  
  ret <4 x i32> %4
}

