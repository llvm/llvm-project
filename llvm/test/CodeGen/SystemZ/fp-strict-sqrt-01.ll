; Test strict 32-bit square root.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)

; Check register square root.
define float @f1(float %val) {
; CHECK-LABEL: f1:
; CHECK: sqebr %f0, %f0
; CHECK: br %r14
  %res = call float @llvm.experimental.constrained.sqrt.f32(
                        float %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

; Check the low end of the SQEB range.
define float @f2(float *%ptr) {
; CHECK-LABEL: f2:
; CHECK: sqeb %f0, 0(%r2)
; CHECK: br %r14
  %val = load float, float *%ptr
  %res = call float @llvm.experimental.constrained.sqrt.f32(
                        float %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

; Check the high end of the aligned SQEB range.
define float @f3(float *%base) {
; CHECK-LABEL: f3:
; CHECK: sqeb %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1023
  %val = load float, float *%ptr
  %res = call float @llvm.experimental.constrained.sqrt.f32(
                        float %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define float @f4(float *%base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: sqeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1024
  %val = load float, float *%ptr
  %res = call float @llvm.experimental.constrained.sqrt.f32(
                        float %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

; Check negative displacements, which also need separate address logic.
define float @f5(float *%base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -4
; CHECK: sqeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 -1
  %val = load float, float *%ptr
  %res = call float @llvm.experimental.constrained.sqrt.f32(
                        float %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

; Check that SQEB allows indices.
define float @f6(float *%base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 2
; CHECK: sqeb %f0, 400(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float, float *%base, i64 %index
  %ptr2 = getelementptr float, float *%ptr1, i64 100
  %val = load float, float *%ptr2
  %res = call float @llvm.experimental.constrained.sqrt.f32(
                        float %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

