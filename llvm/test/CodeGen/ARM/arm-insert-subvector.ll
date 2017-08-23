; RUN: llc -start-before=isel -stop-after=isel -mtriple armv8-unknown-linux -o - < %s | FileCheck %s

define <2 x float> @test_float(<6 x float>* %src) {
  %v= load <6 x float>, <6 x float>* %src, align 1
  %r = shufflevector <6 x float> %v, <6 x float> undef, <2 x i32> <i32 2, i32 5>
  ret <2 x float> %r
}
; CHECK: name: test_float
; CHECK: INSERT_SUBREG

define <2 x i32> @test_i32(<6 x i32>* %src) {
  %v= load <6 x i32>, <6 x i32>* %src, align 1
  %r = shufflevector <6 x i32> %v, <6 x i32> undef, <2 x i32> <i32 2, i32 5>
  ret <2 x i32> %r
}
; CHECK: name: test_i32
; CHECK: INSERT_SUBREG

define <4 x i16> @test_i16(<12 x i16>* %src) {
  %v= load <12 x i16>, <12 x i16>* %src, align 1
  %r = shufflevector <12 x i16> %v, <12 x i16> undef, <4 x i32> <i32 2, i32 5, i32 8, i32 7>
  ret <4 x i16> %r
}
; CHECK: name: test_i16
; CHECK: INSERT_SUBREG

define <8 x i8> @test_i8(<24 x i8>* %src) {
  %v= load <24 x i8>, <24 x i8>* %src, align 1
  %r = shufflevector <24 x i8> %v, <24 x i8> undef, <8 x i32> <i32 2, i32 5, i32 8, i32 11, i32 14, i32 17, i32 20, i32 23>
  ret <8 x i8> %r
}
; CHECK: name: test_i8
; CHECK: INSERT_SUBREG
