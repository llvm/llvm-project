; Test strict extensions of f64 to f128.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare fp128 @llvm.experimental.constrained.fpext.f128.f64(double, metadata)

; Check register extension.
define void @f1(ptr %dst, double %val) #0 {
; CHECK-LABEL: f1:
; CHECK: lxdbr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %val,
                                               metadata !"fpexcept.strict") #0
  store fp128 %res, ptr %dst
  ret void
}

; Check the low end of the LXDB range.
define void @f2(ptr %dst, ptr %ptr) #0 {
; CHECK-LABEL: f2:
; CHECK: lxdb %f0, 0(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %val = load double, ptr %ptr
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %val,
                                               metadata !"fpexcept.strict") #0
  store fp128 %res, ptr %dst
  ret void
}

; Check the high end of the aligned LXDB range.
define void @f3(ptr %dst, ptr %base) #0 {
; CHECK-LABEL: f3:
; CHECK: lxdb %f0, 4088(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 511
  %val = load double, ptr %ptr
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %val,
                                               metadata !"fpexcept.strict") #0
  store fp128 %res, ptr %dst
  ret void
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f4(ptr %dst, ptr %base) #0 {
; CHECK-LABEL: f4:
; CHECK: aghi %r3, 4096
; CHECK: lxdb %f0, 0(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 512
  %val = load double, ptr %ptr
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %val,
                                               metadata !"fpexcept.strict") #0
  store fp128 %res, ptr %dst
  ret void
}

; Check negative displacements, which also need separate address logic.
define void @f5(ptr %dst, ptr %base) #0 {
; CHECK-LABEL: f5:
; CHECK: aghi %r3, -8
; CHECK: lxdb %f0, 0(%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 -1
  %val = load double, ptr %ptr
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %val,
                                               metadata !"fpexcept.strict") #0
  store fp128 %res, ptr %dst
  ret void
}

; Check that LXDB allows indices.
define void @f6(ptr %dst, ptr %base, i64 %index) #0 {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r4, 3
; CHECK: lxdb %f0, 800(%r1,%r3)
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr1 = getelementptr double, ptr %base, i64 %index
  %ptr2 = getelementptr double, ptr %ptr1, i64 100
  %val = load double, ptr %ptr2
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %val,
                                               metadata !"fpexcept.strict") #0
  store fp128 %res, ptr %dst
  ret void
}

attributes #0 = { strictfp }
