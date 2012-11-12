; RUN: llc < %s -march=x86-64 -mcpu=corei7 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i8:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; CHECK: mul_f
define i32 @mul_f(<4 x i8>* %A) {
entry:
; CHECK: pmul
; CHECK-NOT: mulb
  %0 = load <4 x i8>* %A, align 8
  %mul = mul <4 x i8> %0, %0
  store <4 x i8> %mul, <4 x i8>* undef
  ret i32 0
; CHECK: ret
}


; CHECK: shuff_f
define i32 @shuff_f(<4 x i8>* %A) {
entry:
; CHECK: pmovzxbd
; CHECK: paddd
; CHECK: pshufb
  %0 = load <4 x i8>* %A, align 8
  %add = add <4 x i8> %0, %0
  store <4 x i8> %add, <4 x i8>* undef
  ret i32 0
; CHECK: ret
}

; CHECK: bitcast_widen
define <2 x float> @bitcast_widen(<4 x i32> %in) nounwind readnone {
entry:
; CHECK-NOT: pshufd
 %x = shufflevector <4 x i32> %in, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
 %y = bitcast <2 x i32> %x to <2 x float>
 ret <2 x float> %y
; CHECK: ret
}

