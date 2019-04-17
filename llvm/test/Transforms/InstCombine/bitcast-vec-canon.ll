; RUN: opt < %s -instcombine -S | FileCheck %s

define double @a(<1 x i64> %y) {
; CHECK-LABEL: @a(
; CHECK-NEXT:    [[BC:%.*]] = bitcast <1 x i64> %y to <1 x double>
; CHECK-NEXT:    [[C:%.*]] = extractelement <1 x double> [[BC]], i32 0
; CHECK-NEXT:    ret double [[C]]
;
  %c = bitcast <1 x i64> %y to double
  ret double %c
}

define i64 @b(<1 x i64> %y) {
; CHECK-LABEL: @b(
; CHECK-NEXT:    [[TMP1:%.*]] = extractelement <1 x i64> %y, i32 0
; CHECK-NEXT:    ret i64 [[TMP1]]
;
  %c = bitcast <1 x i64> %y to i64
  ret i64 %c
}

define <1 x i64> @c(double %y) {
; CHECK-LABEL: @c(
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast double %y to i64
; CHECK-NEXT:    [[C:%.*]] = insertelement <1 x i64> undef, i64 [[TMP1]], i32 0
; CHECK-NEXT:    ret <1 x i64> [[C]]
;
  %c = bitcast double %y to <1 x i64>
  ret <1 x i64> %c
}

define <1 x i64> @d(i64 %y) {
; CHECK-LABEL: @d(
; CHECK-NEXT:    [[C:%.*]] = insertelement <1 x i64> undef, i64 %y, i32 0
; CHECK-NEXT:    ret <1 x i64> [[C]]
;
  %c = bitcast i64 %y to <1 x i64>
  ret <1 x i64> %c
}

