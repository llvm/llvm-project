; RUN: opt < %s -instcombine -S | FileCheck %s

; Check all scalar / vector combinations for a pair of bitcasts.

define ppc_fp128 @bitcast_bitcast_s_s_s(i128 %a) {
  %bc1 = bitcast i128 %a to fp128
  %bc2 = bitcast fp128 %bc1 to ppc_fp128
  ret ppc_fp128 %bc2

; CHECK-LABEL: @bitcast_bitcast_s_s_s(
; CHECK-NEXT:  %bc2 = bitcast i128 %a to ppc_fp128
; CHECK-NEXT:  ret ppc_fp128 %bc2
}

define <2 x i32> @bitcast_bitcast_s_s_v(i64 %a) {
  %bc1 = bitcast i64 %a to double
  %bc2 = bitcast double %bc1 to <2 x i32>
  ret <2 x i32> %bc2

; CHECK-LABEL: @bitcast_bitcast_s_s_v(
; CHECK-NEXT:  %bc2 = bitcast i64 %a to <2 x i32>
; CHECK-NEXT:  ret <2 x i32> %bc2
}

define double @bitcast_bitcast_s_v_s(i64 %a) {
  %bc1 = bitcast i64 %a to <2 x i32>
  %bc2 = bitcast <2 x i32> %bc1 to double
  ret double %bc2

; CHECK-LABEL: @bitcast_bitcast_s_v_s(
; CHECK-NEXT:  %bc2 = bitcast i64 %a to double
; CHECK-NEXT:  ret double %bc2
}

define <2 x i32> @bitcast_bitcast_s_v_v(i64 %a) {
  %bc1 = bitcast i64 %a to <4 x i16>
  %bc2 = bitcast <4 x i16> %bc1 to <2 x i32>
  ret <2 x i32> %bc2

; CHECK-LABEL: @bitcast_bitcast_s_v_v(
; CHECK-NEXT:  %bc2 = bitcast i64 %a to <2 x i32>
; CHECK-NEXT:  ret <2 x i32> %bc2
}

define i64 @bitcast_bitcast_v_s_s(<2 x i32> %a) {
  %bc1 = bitcast <2 x i32> %a to double
  %bc2 = bitcast double %bc1 to i64
  ret i64 %bc2

; CHECK-LABEL: @bitcast_bitcast_v_s_s(
; CHECK-NEXT:  %bc2 = bitcast <2 x i32> %a to i64
; CHECK-NEXT:  ret i64 %bc2
}

define <4 x i16> @bitcast_bitcast_v_s_v(<2 x i32> %a) {
  %bc1 = bitcast <2 x i32> %a to double
  %bc2 = bitcast double %bc1 to <4 x i16>
  ret <4 x i16> %bc2

; CHECK-LABEL: @bitcast_bitcast_v_s_v(
; CHECK-NEXT:  %bc2 = bitcast <2 x i32> %a to <4 x i16>
; CHECK-NEXT:  ret <4 x i16> %bc2
}

define double @bitcast_bitcast_v_v_s(<2 x float> %a) {
  %bc1 = bitcast <2 x float> %a to <4 x i16>
  %bc2 = bitcast <4 x i16> %bc1 to double
  ret double %bc2

; CHECK-LABEL: @bitcast_bitcast_v_v_s(
; CHECK-NEXT:  %bc2 = bitcast <2 x float> %a to double
; CHECK-NEXT:  ret double %bc2
}

define <2 x i32> @bitcast_bitcast_v_v_v(<2 x float> %a) {
  %bc1 = bitcast <2 x float> %a to <4 x i16>
  %bc2 = bitcast <4 x i16> %bc1 to <2 x i32>
  ret <2 x i32> %bc2

; CHECK-LABEL: @bitcast_bitcast_v_v_v(
; CHECK-NEXT:  %bc2 = bitcast <2 x float> %a to <2 x i32>
; CHECK-NEXT:  ret <2 x i32> %bc2
}

