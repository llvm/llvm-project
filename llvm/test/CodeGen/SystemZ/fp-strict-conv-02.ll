; Test strict extensions of f32 to f64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)

; Check register extension.
define double @f1(float %val) #0 {
; CHECK-LABEL: f1:
; CHECK: ldebr %f0, %f0
; CHECK: br %r14
  %res = call double @llvm.experimental.constrained.fpext.f64.f32(float %val,
                                               metadata !"fpexcept.strict") #0
  ret double %res
}

; Check the low end of the LDEB range.
define double @f2(ptr %ptr) #0 {
; CHECK-LABEL: f2:
; CHECK: ldeb %f0, 0(%r2)
; CHECK: br %r14
  %val = load float, ptr %ptr
  %res = call double @llvm.experimental.constrained.fpext.f64.f32(float %val,
                                               metadata !"fpexcept.strict") #0
  ret double %res
}

; Check the high end of the aligned LDEB range.
define double @f3(ptr %base) #0 {
; CHECK-LABEL: f3:
; CHECK: ldeb %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 1023
  %val = load float, ptr %ptr
  %res = call double @llvm.experimental.constrained.fpext.f64.f32(float %val,
                                               metadata !"fpexcept.strict") #0
  ret double %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(ptr %base) #0 {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: ldeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 1024
  %val = load float, ptr %ptr
  %res = call double @llvm.experimental.constrained.fpext.f64.f32(float %val,
                                               metadata !"fpexcept.strict") #0
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(ptr %base) #0 {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -4
; CHECK: ldeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 -1
  %val = load float, ptr %ptr
  %res = call double @llvm.experimental.constrained.fpext.f64.f32(float %val,
                                               metadata !"fpexcept.strict") #0
  ret double %res
}

; Check that LDEB allows indices.
define double @f6(ptr %base, i64 %index) #0 {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 2
; CHECK: ldeb %f0, 400(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float, ptr %base, i64 %index
  %ptr2 = getelementptr float, ptr %ptr1, i64 100
  %val = load float, ptr %ptr2
  %res = call double @llvm.experimental.constrained.fpext.f64.f32(float %val,
                                               metadata !"fpexcept.strict") #0
  ret double %res
}

attributes #0 = { strictfp }
