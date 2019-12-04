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

define x86_mmx @e(<1 x i64> %y) {
; CHECK-LABEL: @e(
; CHECK-NEXT:    [[TMP1:%.*]] = extractelement <1 x i64> %y, i32 0
; CHECK-NEXT:    [[C:%.*]] = bitcast i64 [[TMP1]] to x86_mmx
; CHECK-NEXT:    ret x86_mmx [[C]]
;
  %c = bitcast <1 x i64> %y to x86_mmx
  ret x86_mmx %c
}

define <1 x i64> @f(x86_mmx %y) {
; CHECK-LABEL: @f(
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast x86_mmx %y to i64
; CHECK-NEXT:    [[C:%.*]] = insertelement <1 x i64> undef, i64 [[TMP1]], i32 0
; CHECK-NEXT:    ret <1 x i64> [[C]]
;
  %c = bitcast x86_mmx %y to <1 x i64>
  ret <1 x i64> %c
}

define double @g(x86_mmx %x) {
; CHECK-LABEL: @g(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast x86_mmx %x to double
; CHECK-NEXT:    ret double [[TMP0]]
;
entry:
  %0 = bitcast x86_mmx %x to <1 x i64>
  %1 = bitcast <1 x i64> %0 to double
  ret double %1
}
