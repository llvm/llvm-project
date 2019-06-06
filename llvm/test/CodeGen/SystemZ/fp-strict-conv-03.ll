; Test strict extensions of f32 to f128.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare fp128 @llvm.experimental.constrained.fpext.f128.f32(float, metadata)

; Check register extension.
define void @f1(fp128 *%dst, float %val) {
; CHECK-LABEL: f1:
; CHECK: lxebr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f32(float %val,
                                               metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%dst
  ret void
}

; Check extension from memory.
; FIXME: This should really use LXEB, but there is no strict "extload" yet.
define void @f2(fp128 *%dst, float *%ptr) {
; CHECK-LABEL: f2:
; CHECK: le %f0, 0(%r3)
; CHECK: lxebr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %val = load float, float *%ptr
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f32(float %val,
                                               metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%dst
  ret void
}

