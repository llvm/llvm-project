; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; X * Y (when Y is a boolean) --> Y ? X : 0

define float @fmul_bool(float %x, i1 %y) {
; CHECK-LABEL: @fmul_bool(
; CHECK-NEXT:    [[M:%.*]] = select nnan nsz i1 [[Y:%.*]], float [[X:%.*]], float 0.000000e+00
; CHECK-NEXT:    ret float [[M]]
;
  %z = uitofp i1 %y to float
  %m = fmul nnan nsz float %z, %x
  ret float %m
}

define <2 x float> @fmul_bool_vec(<2 x float> %x, <2 x i1> %y) {
; CHECK-LABEL: @fmul_bool_vec(
; CHECK-NEXT:    [[M:%.*]] = select nnan nsz <2 x i1> [[Y:%.*]], <2 x float> [[X:%.*]], <2 x float> zeroinitializer
; CHECK-NEXT:    ret <2 x float> [[M]]
;
  %z = uitofp <2 x i1> %y to <2 x float>
  %m = fmul nnan nsz <2 x float> %z, %x
  ret <2 x float> %m
}

define <2 x float> @fmul_bool_vec_commute(<2 x float> %px, <2 x i1> %y) {
; CHECK-LABEL: @fmul_bool_vec_commute(
; CHECK-NEXT:    [[X:%.*]] = fmul nnan nsz <2 x float> [[PX:%.*]], [[PX]]
; CHECK-NEXT:    [[M:%.*]] = select nnan nsz <2 x i1> [[Y:%.*]], <2 x float> [[X]], <2 x float> zeroinitializer
; CHECK-NEXT:    ret <2 x float> [[M]]
;
  %x = fmul nnan nsz <2 x float> %px, %px  ; thwart complexity-based canonicalization
  %z = uitofp <2 x i1> %y to <2 x float>
  %m = fmul nnan nsz <2 x float> %x, %z
  ret <2 x float> %m
}
