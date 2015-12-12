; RUN: opt < %s -instcombine -S | FileCheck %s

define double @a(<1 x i64> %y) {
  %c = bitcast <1 x i64> %y to double
  ret double %c
 
; CHECK-LABEL: @a(
; CHECK-NEXT:  extractelement <1 x i64> %y, i32 0
; CHECK-NEXT:  bitcast i64 {{.*}} to double
; CHECK-NEXT:  ret double
}

define i64 @b(<1 x i64> %y) {
  %c = bitcast <1 x i64> %y to i64
  ret i64 %c

; CHECK-LABEL: @b(
; CHECK-NEXT:  extractelement <1 x i64> %y, i32 0
; CHECK-NEXT:  ret i64
}

define <1 x i64> @c(double %y) {
  %c = bitcast double %y to <1 x i64>
  ret <1 x i64> %c

; CHECK-LABEL: @c(
; CHECK-NEXT:  bitcast double %y to i64
; CHECK-NEXT:  insertelement <1 x i64> undef, i64 {{.*}}, i32 0
; CHECK-NEXT:  ret <1 x i64>
}

define <1 x i64> @d(i64 %y) {
  %c = bitcast i64 %y to <1 x i64>
  ret <1 x i64> %c

; CHECK-LABEL: @d(
; CHECK-NEXT:  insertelement <1 x i64> undef, i64 %y, i32 0
; CHECK-NEXT:  ret <1 x i64>
}


