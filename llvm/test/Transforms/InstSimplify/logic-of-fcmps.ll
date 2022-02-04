; RUN: opt < %s -passes=instsimplify -S | FileCheck %s

; Cycle through commuted variants where one operand of fcmp ord/uno is
; known not-a-NAN and the other is repeated in the logically-connected fcmp.

define i1 @ord1(float %x, float %y) {
; CHECK-LABEL: @ord1(
; CHECK-NEXT:    [[CMP2:%.*]] = fcmp ord float %x, %y
; CHECK-NEXT:    ret i1 [[CMP2]]
;
  %cmp1 = fcmp ord float 0.0, %x
  %cmp2 = fcmp ord float %x, %y
  %r = and i1 %cmp1, %cmp2
  ret i1 %r
}

define i1 @ord2(double %x, double %y) {
; CHECK-LABEL: @ord2(
; CHECK-NEXT:    [[CMP2:%.*]] = fcmp ord double %y, %x
; CHECK-NEXT:    ret i1 [[CMP2]]
;
  %cmp1 = fcmp ord double 42.0, %x
  %cmp2 = fcmp ord double %y, %x
  %r = and i1 %cmp1, %cmp2
  ret i1 %r
}

define <2 x i1> @ord3(<2 x float> %x, <2 x float> %y) {
; CHECK-LABEL: @ord3(
; CHECK-NEXT:    [[CMP2:%.*]] = fcmp ord <2 x float> %x, %y
; CHECK-NEXT:    ret <2 x i1> [[CMP2]]
;
  %cmp1 = fcmp ord <2 x float> %x, zeroinitializer
  %cmp2 = fcmp ord <2 x float> %x, %y
  %r = and <2 x i1> %cmp1, %cmp2
  ret <2 x i1> %r
}

define <2 x i1> @ord4(<2 x double> %x, <2 x double> %y) {
; CHECK-LABEL: @ord4(
; CHECK-NEXT:    [[CMP2:%.*]] = fcmp ord <2 x double> %y, %x
; CHECK-NEXT:    ret <2 x i1> [[CMP2]]
;
  %cmp1 = fcmp ord <2 x double> %x, <double 42.0, double 42.0>
  %cmp2 = fcmp ord <2 x double> %y, %x
  %r = and <2 x i1> %cmp1, %cmp2
  ret <2 x i1> %r
}

define i1 @ord5(float %x, float %y) {
; CHECK-LABEL: @ord5(
; CHECK-NEXT:    [[CMP1:%.*]] = fcmp ord float %x, %y
; CHECK-NEXT:    ret i1 [[CMP1]]
;
  %nnan = fdiv nnan float %x, %y
  %cmp1 = fcmp ord float %x, %y
  %cmp2 = fcmp ord float %nnan, %x
  %r = and i1 %cmp1, %cmp2
  ret i1 %r
}

define i1 @ord6(double %x, double %y) {
; CHECK-LABEL: @ord6(
; CHECK-NEXT:    [[CMP1:%.*]] = fcmp ord double %y, %x
; CHECK-NEXT:    ret i1 [[CMP1]]
;
  %cmp1 = fcmp ord double %y, %x
  %cmp2 = fcmp ord double 42.0, %x
  %r = and i1 %cmp1, %cmp2
  ret i1 %r
}

define <2 x i1> @ord7(<2 x float> %x, <2 x float> %y) {
; CHECK-LABEL: @ord7(
; CHECK-NEXT:    [[CMP1:%.*]] = fcmp ord <2 x float> %x, %y
; CHECK-NEXT:    ret <2 x i1> [[CMP1]]
;
  %cmp1 = fcmp ord <2 x float> %x, %y
  %cmp2 = fcmp ord <2 x float> %x, zeroinitializer
  %r = and <2 x i1> %cmp1, %cmp2
  ret <2 x i1> %r
}

define <2 x i1> @ord8(<2 x double> %x, <2 x double> %y) {
; CHECK-LABEL: @ord8(
; CHECK-NEXT:    [[CMP1:%.*]] = fcmp ord <2 x double> %y, %x
; CHECK-NEXT:    ret <2 x i1> [[CMP1]]
;
  %cmp1 = fcmp ord <2 x double> %y, %x
  %cmp2 = fcmp ord <2 x double> %x, <double 0.0, double 42.0>
  %r = and <2 x i1> %cmp1, %cmp2
  ret <2 x i1> %r
}

define i1 @uno1(float %x, float %y) {
; CHECK-LABEL: @uno1(
; CHECK-NEXT:    [[CMP2:%.*]] = fcmp uno float %x, %y
; CHECK-NEXT:    ret i1 [[CMP2]]
;
  %cmp1 = fcmp uno float 0.0, %x
  %cmp2 = fcmp uno float %x, %y
  %r = or i1 %cmp1, %cmp2
  ret i1 %r
}

define i1 @uno2(double %x, double %y) {
; CHECK-LABEL: @uno2(
; CHECK-NEXT:    [[CMP2:%.*]] = fcmp uno double %y, %x
; CHECK-NEXT:    ret i1 [[CMP2]]
;
  %cmp1 = fcmp uno double 42.0, %x
  %cmp2 = fcmp uno double %y, %x
  %r = or i1 %cmp1, %cmp2
  ret i1 %r
}

define <2 x i1> @uno3(<2 x float> %x, <2 x float> %y) {
; CHECK-LABEL: @uno3(
; CHECK-NEXT:    [[CMP2:%.*]] = fcmp uno <2 x float> %x, %y
; CHECK-NEXT:    ret <2 x i1> [[CMP2]]
;
  %cmp1 = fcmp uno <2 x float> %x, zeroinitializer
  %cmp2 = fcmp uno <2 x float> %x, %y
  %r = or <2 x i1> %cmp1, %cmp2
  ret <2 x i1> %r
}

define <2 x i1> @uno4(<2 x double> %x, <2 x double> %y) {
; CHECK-LABEL: @uno4(
; CHECK-NEXT:    [[CMP2:%.*]] = fcmp uno <2 x double> %y, %x
; CHECK-NEXT:    ret <2 x i1> [[CMP2]]
;
  %cmp1 = fcmp uno <2 x double> %x, <double 42.0, double 42.0>
  %cmp2 = fcmp uno <2 x double> %y, %x
  %r = or <2 x i1> %cmp1, %cmp2
  ret <2 x i1> %r
}

define i1 @uno5(float %x, float %y) {
; CHECK-LABEL: @uno5(
; CHECK-NEXT:    [[CMP1:%.*]] = fcmp uno float %x, %y
; CHECK-NEXT:    ret i1 [[CMP1]]
;
  %cmp1 = fcmp uno float %x, %y
  %cmp2 = fcmp uno float 0.0, %x
  %r = or i1 %cmp1, %cmp2
  ret i1 %r
}

define i1 @uno6(double %x, double %y) {
; CHECK-LABEL: @uno6(
; CHECK-NEXT:    [[CMP1:%.*]] = fcmp uno double %y, %x
; CHECK-NEXT:    ret i1 [[CMP1]]
;
  %cmp1 = fcmp uno double %y, %x
  %cmp2 = fcmp uno double 42.0, %x
  %r = or i1 %cmp1, %cmp2
  ret i1 %r
}

define <2 x i1> @uno7(<2 x float> %x, <2 x float> %y) {
; CHECK-LABEL: @uno7(
; CHECK-NEXT:    [[CMP1:%.*]] = fcmp uno <2 x float> %x, %y
; CHECK-NEXT:    ret <2 x i1> [[CMP1]]
;
  %nnan = fdiv nnan <2 x float> %x, %y
  %cmp1 = fcmp uno <2 x float> %x, %y
  %cmp2 = fcmp uno <2 x float> %x, %nnan
  %r = or <2 x i1> %cmp1, %cmp2
  ret <2 x i1> %r
}

define <2 x i1> @uno8(<2 x double> %x, <2 x double> %y) {
; CHECK-LABEL: @uno8(
; CHECK-NEXT:    [[CMP1:%.*]] = fcmp uno <2 x double> %y, %x
; CHECK-NEXT:    ret <2 x i1> [[CMP1]]
;
  %cmp1 = fcmp uno <2 x double> %y, %x
  %cmp2 = fcmp uno <2 x double> %x, <double 0x7ff0000000000000, double 42.0>
  %r = or <2 x i1> %cmp1, %cmp2
  ret <2 x i1> %r
}

