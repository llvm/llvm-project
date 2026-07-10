; RUN: opt -passes=instsimplify -S < %s | FileCheck %s

; Negative test: a shufflevector with a non-constant (dynamic) mask must not
; be simplified away -- InstructionSimplify has no dynamic-mask folds yet.
define <4 x i32> @dynamic_mask(<4 x i32> %v1, <4 x i32> %v2, <4 x i32> %mask) {
; CHECK-LABEL: @dynamic_mask(
; CHECK-NEXT:    [[RES:%.*]] = shufflevector <4 x i32> [[V1:%.*]], <4 x i32> [[V2:%.*]], <4 x i32> [[MASK:%.*]]
; CHECK-NEXT:    ret <4 x i32> [[RES]]
;
  %res = shufflevector <4 x i32> %v1, <4 x i32> %v2, <4 x i32> %mask
  ret <4 x i32> %res
}

define <4 x i32> @poison_operands_dynamic_mask(<4 x i32> %mask) {
; CHECK-LABEL: @poison_operands_dynamic_mask(
; CHECK-NEXT:    [[RES:%.*]] = shufflevector <4 x i32> poison, <4 x i32> poison, <4 x i32> [[MASK:%.*]]
; CHECK-NEXT:    ret <4 x i32> [[RES]]
;
  %res = shufflevector <4 x i32> poison, <4 x i32> poison, <4 x i32> %mask
  ret <4 x i32> %res
}
