; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: @a
; CHECK-NOT: bitcast
; CHECK: ret
define <4 x i32> @a(<1 x i64> %y) {
  %c = bitcast <2 x i64> <i64 0, i64 0> to <4 x i32>
  ret <4 x i32> %c
}

; CHECK: @b
; CHECK-NOT: bitcast
; CHECK: ret

define <4 x i32> @b(<1 x i64> %y) {
  %c = bitcast <2 x i64> <i64 -1, i64 -1> to <4 x i32>
  ret <4 x i32> %c
}

; CHECK: @foo
; CHECK-NOT: bitcast
; CHECK: ret

; from MultiSource/Benchmarks/Bullet
define <2 x float> @foo() {
  %cast = bitcast i64 -1 to <2 x float>
  ret <2 x float> %cast
}


; CHECK: @foo2
; CHECK-NOT: bitcast
; CHECK: ret
define <2 x double> @foo2() {
  %cast = bitcast i128 -1 to <2 x double>
  ret <2 x double> %cast
}

; CHECK: @foo3
; CHECK-NOT: bitcast
; CHECK: ret
define <1 x float> @foo3() {
  %cast = bitcast i32 -1 to <1 x float>
  ret <1 x float> %cast
}

; CHECK: @foo4
; CHECK-NOT: bitcast
; CHECK: ret
define float @foo4() {
  %cast = bitcast <1 x i32 ><i32 -1> to float
  ret float %cast
}

; CHECK: @foo5
; CHECK-NOT: bitcast
; CHECK: ret
define double @foo5() {
  %cast = bitcast <2 x i32 ><i32 -1, i32 -1> to double
  ret double %cast
}


; CHECK: @foo6
; CHECK-NOT: bitcast
; CHECK: ret
define <2 x double> @foo6() {
  %cast = bitcast <4 x i32><i32 -1, i32 -1, i32 -1, i32 -1> to <2 x double>
  ret <2 x double> %cast
}
