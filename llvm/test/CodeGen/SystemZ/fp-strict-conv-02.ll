; Test strict extensions of f32 to f64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-VECTOR %s

declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)

; Check register extension.
define double @f1(float %val) {
; CHECK-LABEL: f1:
; CHECK: ldebr %f0, %f0
; CHECK: br %r14
  %res = call double @llvm.experimental.constrained.fpext.f64.f32(float %val,
                                               metadata !"fpexcept.strict")
  ret double %res
}

; Check extension from memory.
; FIXME: This should really use LDEB, but there is no strict "extload" yet.
define double @f2(float *%ptr) {
; CHECK-LABEL: f2:
; CHECK-SCALAR: le %f0, 0(%r2)
; CHECK-VECTOR: lde %f0, 0(%r2)
; CHECK: ldebr %f0, %f0
; CHECK: br %r14
  %val = load float, float *%ptr
  %res = call double @llvm.experimental.constrained.fpext.f64.f32(float %val,
                                               metadata !"fpexcept.strict")
  ret double %res
}

