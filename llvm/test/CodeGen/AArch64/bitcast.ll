; RUN: llc < %s -mtriple=aarch64--linux-gnu | FileCheck %s

; PR23065: SCALAR_TO_VECTOR implies the top elements 1 to N-1 of the N-element vector are undefined.

define <4 x i16> @foo1(<2 x i32> %a) {
; CHECK-LABEL: foo1:
; CHECK:       movi	v0.2d, #0000000000000000
; CHECK-NEXT:  ret

  %1 = shufflevector <2 x i32> <i32 58712, i32 undef>, <2 x i32> %a, <2 x i32> <i32 0, i32 2>
; Can't optimize the following bitcast to scalar_to_vector.
  %2 = bitcast <2 x i32> %1 to <4 x i16>
  %3 = shufflevector <4 x i16> %2, <4 x i16> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  ret <4 x i16> %3
}

define <4 x i16> @foo2(<2 x i32> %a) {
; CHECK-LABEL: foo2:
; CHECK:       movi	v0.2d, #0000000000000000
; CHECK-NEXT:  ret

  %1 = shufflevector <2 x i32> <i32 712, i32 undef>, <2 x i32> %a, <2 x i32> <i32 0, i32 2>
; Can't optimize the following bitcast to scalar_to_vector.
  %2 = bitcast <2 x i32> %1 to <4 x i16>
  %3 = shufflevector <4 x i16> %2, <4 x i16> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  ret <4 x i16> %3
}
