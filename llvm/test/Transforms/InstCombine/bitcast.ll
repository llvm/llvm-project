; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; Bitcasts between vectors and scalars are valid.
; PR4487
define i32 @test1(i64 %a) {
        %t1 = bitcast i64 %a to <2 x i32>
        %t2 = bitcast i64 %a to <2 x i32>
        %t3 = xor <2 x i32> %t1, %t2
        %t4 = extractelement <2 x i32> %t3, i32 0
        ret i32 %t4

; CHECK-LABEL: @test1(
; CHECK: ret i32 0
}

; Perform the bitwise logic in the source type of the operands to eliminate bitcasts.

define <2 x i32> @xor_two_vector_bitcasts(<1 x i64> %a, <1 x i64> %b) {
  %t1 = bitcast <1 x i64> %a to <2 x i32>
  %t2 = bitcast <1 x i64> %b to <2 x i32>
  %t3 = xor <2 x i32> %t1, %t2
  ret <2 x i32> %t3

; CHECK-LABEL: @xor_two_vector_bitcasts(
; CHECK-NEXT:  %t31 = xor <1 x i64> %a, %b
; CHECK-NEXT:  %t3 = bitcast <1 x i64> %t31 to <2 x i32>
; CHECK-NEXT:  ret <2 x i32> %t3
}

; Verify that 'xor' of vector and constant is done as a vector bitwise op before the bitcast.

define <2 x i32> @xor_bitcast_vec_to_vec(<1 x i64> %a) {
  %t1 = bitcast <1 x i64> %a to <2 x i32>
  %t2 = xor <2 x i32> <i32 1, i32 2>, %t1
  ret <2 x i32> %t2

; CHECK-LABEL: @xor_bitcast_vec_to_vec(
; CHECK-NEXT:  %t21 = xor <1 x i64> %a, <i64 8589934593> 
; CHECK-NEXT:  %t2 = bitcast <1 x i64> %t21 to <2 x i32>
; CHECK-NEXT:  ret <2 x i32> %t2
}

; Verify that 'and' of integer and constant is done as a vector bitwise op before the bitcast.

define i64 @and_bitcast_vec_to_int(<2 x i32> %a) {
  %t1 = bitcast <2 x i32> %a to i64
  %t2 = and i64 %t1, 3
  ret i64 %t2

; CHECK-LABEL: @and_bitcast_vec_to_int(
; CHECK-NEXT:  %t21 = and <2 x i32> %a, <i32 3, i32 0>
; CHECK-NEXT:  %t2 = bitcast <2 x i32> %t21 to i64
; CHECK-NEXT:  ret i64 %t2
}

; Verify that 'or' of vector and constant is done as an integer bitwise op before the bitcast.

define <2 x i32> @or_bitcast_int_to_vec(i64 %a) {
  %t1 = bitcast i64 %a to <2 x i32>
  %t2 = or <2 x i32> %t1, <i32 1, i32 2>
  ret <2 x i32> %t2

; CHECK-LABEL: @or_bitcast_int_to_vec(
; CHECK-NEXT:  %t21 = or i64 %a, 8589934593
; CHECK-NEXT:  %t2 = bitcast i64 %t21 to <2 x i32>
; CHECK-NEXT:  ret <2 x i32> %t2
}

; Optimize bitcasts that are extracting low element of vector.  This happens
; because of SRoA.
; rdar://7892780
define float @test2(<2 x float> %A, <2 x i32> %B) {
  %tmp28 = bitcast <2 x float> %A to i64  ; <i64> [#uses=2]
  %tmp23 = trunc i64 %tmp28 to i32                ; <i32> [#uses=1]
  %tmp24 = bitcast i32 %tmp23 to float            ; <float> [#uses=1]

  %tmp = bitcast <2 x i32> %B to i64
  %tmp2 = trunc i64 %tmp to i32                ; <i32> [#uses=1]
  %tmp4 = bitcast i32 %tmp2 to float            ; <float> [#uses=1]

  %add = fadd float %tmp24, %tmp4
  ret float %add

; CHECK-LABEL: @test2(
; CHECK-NEXT:  %tmp24 = extractelement <2 x float> %A, i32 0
; CHECK-NEXT:  bitcast <2 x i32> %B to <2 x float>
; CHECK-NEXT:  %tmp4 = extractelement <2 x float> {{.*}}, i32 0
; CHECK-NEXT:  %add = fadd float %tmp24, %tmp4
; CHECK-NEXT:  ret float %add
}

; Optimize bitcasts that are extracting other elements of a vector.  This
; happens because of SRoA.
; rdar://7892780
define float @test3(<2 x float> %A, <2 x i64> %B) {
  %tmp28 = bitcast <2 x float> %A to i64
  %tmp29 = lshr i64 %tmp28, 32
  %tmp23 = trunc i64 %tmp29 to i32
  %tmp24 = bitcast i32 %tmp23 to float

  %tmp = bitcast <2 x i64> %B to i128
  %tmp1 = lshr i128 %tmp, 64
  %tmp2 = trunc i128 %tmp1 to i32
  %tmp4 = bitcast i32 %tmp2 to float

  %add = fadd float %tmp24, %tmp4
  ret float %add

; CHECK-LABEL: @test3(
; CHECK-NEXT:  %tmp24 = extractelement <2 x float> %A, i32 1
; CHECK-NEXT:  bitcast <2 x i64> %B to <4 x float>
; CHECK-NEXT:  %tmp4 = extractelement <4 x float> {{.*}}, i32 2
; CHECK-NEXT:  %add = fadd float %tmp24, %tmp4
; CHECK-NEXT:  ret float %add
}

; Both bitcasts are unnecessary; change the extractelement.

define float @bitcast_extelt1(<2 x float> %A) {
  %bc1 = bitcast <2 x float> %A to <2 x i32>
  %ext = extractelement <2 x i32> %bc1, i32 0
  %bc2 = bitcast i32 %ext to float
  ret float %bc2

; CHECK-LABEL: @bitcast_extelt1(
; CHECK-NEXT:  %bc2 = extractelement <2 x float> %A, i32 0
; CHECK-NEXT:  ret float %bc2
}

; Second bitcast can be folded into the first.

define i64 @bitcast_extelt2(<4 x float> %A) {
  %bc1 = bitcast <4 x float> %A to <2 x double>
  %ext = extractelement <2 x double> %bc1, i32 1
  %bc2 = bitcast double %ext to i64
  ret i64 %bc2

; CHECK-LABEL: @bitcast_extelt2(
; CHECK-NEXT:  %bc = bitcast <4 x float> %A to <2 x i64>
; CHECK-NEXT:  %bc2 = extractelement <2 x i64> %bc, i32 1
; CHECK-NEXT:  ret i64 %bc2
}

; TODO: This should return %A. 

define <2 x i32> @bitcast_extelt3(<2 x i32> %A) {
  %bc1 = bitcast <2 x i32> %A to <1 x i64>
  %ext = extractelement <1 x i64> %bc1, i32 0
  %bc2 = bitcast i64 %ext to <2 x i32>
  ret <2 x i32> %bc2

; CHECK-LABEL: @bitcast_extelt3(
; CHECK-NEXT:  %bc1 = bitcast <2 x i32> %A to <1 x i64>
; CHECK-NEXT:  %ext = extractelement <1 x i64> %bc1, i32 0
; CHECK-NEXT:  %bc2 = bitcast i64 %ext to <2 x i32>
; CHECK-NEXT:  ret <2 x i32> %bc2
}

; Handle the case where the input is not a vector.

define double @bitcast_extelt4(i128 %A) {
  %bc1 = bitcast i128 %A to <2 x i64>
  %ext = extractelement <2 x i64> %bc1, i32 0
  %bc2 = bitcast i64 %ext to double
  ret double %bc2

; CHECK-LABEL: @bitcast_extelt4(
; CHECK-NEXT:  %bc = bitcast i128 %A to <2 x double>
; CHECK-NEXT:  %bc2 = extractelement <2 x double> %bc, i32 0
; CHECK-NEXT:  ret double %bc2
}

define <2 x i32> @test4(i32 %A, i32 %B){
  %tmp38 = zext i32 %A to i64
  %tmp32 = zext i32 %B to i64
  %tmp33 = shl i64 %tmp32, 32
  %ins35 = or i64 %tmp33, %tmp38
  %tmp43 = bitcast i64 %ins35 to <2 x i32>
  ret <2 x i32> %tmp43
  ; CHECK-LABEL: @test4(
  ; CHECK-NEXT: insertelement <2 x i32> undef, i32 %A, i32 0
  ; CHECK-NEXT: insertelement <2 x i32> {{.*}}, i32 %B, i32 1
  ; CHECK-NEXT: ret <2 x i32>

}

; rdar://8360454
define <2 x float> @test5(float %A, float %B) {
  %tmp37 = bitcast float %A to i32
  %tmp38 = zext i32 %tmp37 to i64
  %tmp31 = bitcast float %B to i32
  %tmp32 = zext i32 %tmp31 to i64
  %tmp33 = shl i64 %tmp32, 32
  %ins35 = or i64 %tmp33, %tmp38
  %tmp43 = bitcast i64 %ins35 to <2 x float>
  ret <2 x float> %tmp43
  ; CHECK-LABEL: @test5(
  ; CHECK-NEXT: insertelement <2 x float> undef, float %A, i32 0
  ; CHECK-NEXT: insertelement <2 x float> {{.*}}, float %B, i32 1
  ; CHECK-NEXT: ret <2 x float>
}

define <2 x float> @test6(float %A){
  %tmp23 = bitcast float %A to i32              ; <i32> [#uses=1]
  %tmp24 = zext i32 %tmp23 to i64                 ; <i64> [#uses=1]
  %tmp25 = shl i64 %tmp24, 32                     ; <i64> [#uses=1]
  %mask20 = or i64 %tmp25, 1109917696             ; <i64> [#uses=1]
  %tmp35 = bitcast i64 %mask20 to <2 x float>     ; <<2 x float>> [#uses=1]
  ret <2 x float> %tmp35
; CHECK-LABEL: @test6(
; CHECK-NEXT: insertelement <2 x float> <float 4.200000e+01, float undef>, float %A, i32 1
; CHECK: ret
}

define i64 @ISPC0(i64 %in) {
  %out = and i64 %in, xor (i64 bitcast (<4 x i16> <i16 -1, i16 -1, i16 -1, i16 -1> to i64), i64 -1)
  ret i64 %out
; CHECK-LABEL: @ISPC0(
; CHECK: ret i64 0
}


define i64 @Vec2(i64 %in) {
  %out = and i64 %in, xor (i64 bitcast (<4 x i16> <i16 0, i16 0, i16 0, i16 0> to i64), i64 0)
  ret i64 %out
; CHECK-LABEL: @Vec2(
; CHECK: ret i64 0
}

define i64 @All11(i64 %in) {
  %out = and i64 %in, xor (i64 bitcast (<2 x float> bitcast (i64 -1 to <2 x float>) to i64), i64 -1)
  ret i64 %out
; CHECK-LABEL: @All11(
; CHECK: ret i64 0
}


define i32 @All111(i32 %in) {
  %out = and i32 %in, xor (i32 bitcast (<1 x float> bitcast (i32 -1 to <1 x float>) to i32), i32 -1)
  ret i32 %out
; CHECK-LABEL: @All111(
; CHECK: ret i32 0
}

define <2 x i16> @BitcastInsert(i32 %a) {
  %v = insertelement <1 x i32> undef, i32 %a, i32 0
  %r = bitcast <1 x i32> %v to <2 x i16>
  ret <2 x i16> %r
; CHECK-LABEL: @BitcastInsert(
; CHECK: bitcast i32 %a to <2 x i16>
}

; PR17293
define <2 x i64> @test7(<2 x i8*>* %arg) nounwind {
  %cast = bitcast <2 x i8*>* %arg to <2 x i64>*
  %load = load <2 x i64>, <2 x i64>* %cast, align 16
  ret <2 x i64> %load
; CHECK: @test7
; CHECK: bitcast
; CHECK: load
}

define i8 @test8() {
  %res = bitcast <8 x i1> <i1 true, i1 true, i1 false, i1 true, i1 false, i1 true, i1 false, i1 true> to i8
  ret i8 %res
; CHECK: @test8
; CHECK: ret i8 -85
}
