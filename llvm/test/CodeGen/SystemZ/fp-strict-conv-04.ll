; Test strict extensions of f64 to f128.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare fp128 @llvm.experimental.constrained.fpext.f128.f64(double, metadata)

; Check register extension.
define void @f1(fp128 *%dst, double %val) {
; CHECK-LABEL: f1:
; CHECK: lxdbr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %val,
                                               metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%dst
  ret void
}

; Check extension from memory.
; FIXME: This should really use LXDB, but there is no strict "extload" yet.
define void @f2(fp128 *%dst, double *%ptr) {
; CHECK-LABEL: f2:
; CHECK: ld %f0, 0(%r3)
; CHECK: lxdbr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %val = load double, double *%ptr
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %val,
                                               metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%dst
  ret void
}

