; RUN: opt -mtriple=amdgcn-amd-amdhsa -amdgpu-codegenprepare -verify -S %s -o - | FileCheck %s

declare i1 @llvm.amdgcn.class.f32(float, i32) nounwind readnone
declare i1 @llvm.amdgcn.class.f64(double, i32) nounwind readnone

; Trivial case, xor instruction should be removed and
; the second argument of the intrinsic call should be
; bitwise-negated
; CHECK: @fold_negate_intrinsic_test_mask
; CHECK: %1 = call i1 @llvm.amdgcn.class.f32(float %x, i32 1018)
define i1 @fold_negate_intrinsic_test_mask(float %x) nounwind {
  %1 = call i1 @llvm.amdgcn.class.f32(float %x, i32 5)
  %2 = xor i1 %1, -1
  ret i1 %2
}

; Trivial case, xor instruction should be removed and
; the second argument of the intrinsic call should be
; bitwise-negated
; CHECK: @fold_negate_intrinsic_test_mask_dbl
; CHECK: %1 = call i1 @llvm.amdgcn.class.f64(double %x, i32 1018)
define i1 @fold_negate_intrinsic_test_mask_dbl(double %x) nounwind {
  %1 = call i1 @llvm.amdgcn.class.f64(double %x, i32 5)
  %2 = xor i1 %1, -1
  ret i1 %2
}

; Negative test: should not transform for variable test masks
; CHECK: @fold_negate_intrinsic_test_mask_neg_var
; CHECK: %[[X0:.*]] = alloca i32
; CHECK: %[[X1:.*]] = load i32, i32* %[[X0]]
; CHECK: call i1 @llvm.amdgcn.class.f32(float %x, i32 %[[X1]])
; CHECK: xor
define i1 @fold_negate_intrinsic_test_mask_neg_var(float %x) nounwind {
  %1 = alloca i32
  store i32 7, i32* %1
  %2 = load i32, i32* %1
  %3 = call i1 @llvm.amdgcn.class.f32(float %x, i32 %2)
  %4 = xor i1 %3, -1
  ret i1 %4
}

; Negative test: should not transform for multiple uses of the
;   intrinsic returned value
; CHECK: @fold_negate_intrinsic_test_mask_neg_multiple_uses
; CHECK: %[[X1:.*]] = call i1 @llvm.amdgcn.class.f32(float %x, i32 7)
; CHECK: store i1 %[[X1]]
; CHECK: %[[X2:.*]] = xor i1 %[[X1]]
define i1 @fold_negate_intrinsic_test_mask_neg_multiple_uses(float %x) nounwind {
  %y = alloca i1
  %1 = call i1 @llvm.amdgcn.class.f32(float %x, i32 7)
  %2 = xor i1 %1, -1
  store i1 %1, i1* %y
  %3 = xor i1 %1, -1
  ret i1 %2
}

; Negative test: should not transform for a xor with no operand equal to -1
; CHECK: @fold_negate_intrinsic_test_mask_neg_one
; CHECK: %[[X0:.*]] = call i1 @llvm.amdgcn.class.f32(float %x, i32 7)
; CHECK: xor i1 %[[X0]], false
define i1 @fold_negate_intrinsic_test_mask_neg_one(float %x) nounwind {
  %1 = call i1 @llvm.amdgcn.class.f32(float %x, i32 7)
  %2 = xor i1 %1, false
  ret i1 %2
}
