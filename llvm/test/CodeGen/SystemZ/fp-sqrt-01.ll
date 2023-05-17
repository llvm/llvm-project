; Test 32-bit square root.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @llvm.sqrt.f32(float)
declare float @sqrtf(float)

; Check register square root.
define float @f1(float %val) {
; CHECK-LABEL: f1:
; CHECK: sqebr %f0, %f0
; CHECK: br %r14
  %res = call float @llvm.sqrt.f32(float %val)
  ret float %res
}

; Check the low end of the SQEB range.
define float @f2(ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: sqeb %f0, 0(%r2)
; CHECK: br %r14
  %val = load float, ptr %ptr
  %res = call float @llvm.sqrt.f32(float %val)
  ret float %res
}

; Check the high end of the aligned SQEB range.
define float @f3(ptr %base) {
; CHECK-LABEL: f3:
; CHECK: sqeb %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 1023
  %val = load float, ptr %ptr
  %res = call float @llvm.sqrt.f32(float %val)
  ret float %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define float @f4(ptr %base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: sqeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 1024
  %val = load float, ptr %ptr
  %res = call float @llvm.sqrt.f32(float %val)
  ret float %res
}

; Check negative displacements, which also need separate address logic.
define float @f5(ptr %base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -4
; CHECK: sqeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 -1
  %val = load float, ptr %ptr
  %res = call float @llvm.sqrt.f32(float %val)
  ret float %res
}

; Check that SQEB allows indices.
define float @f6(ptr %base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 2
; CHECK: sqeb %f0, 400(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float, ptr %base, i64 %index
  %ptr2 = getelementptr float, ptr %ptr1, i64 100
  %val = load float, ptr %ptr2
  %res = call float @llvm.sqrt.f32(float %val)
  ret float %res
}

; Test a case where we spill the source of at least one SQEBR.  We want
; to use SQEB if possible.
define void @f7(ptr %ptr) {
; CHECK-LABEL: f7:
; CHECK-SCALAR: sqeb {{%f[0-9]+}}, 16{{[04]}}(%r15)
; CHECK: br %r14
  %val0 = load volatile float, ptr %ptr
  %val1 = load volatile float, ptr %ptr
  %val2 = load volatile float, ptr %ptr
  %val3 = load volatile float, ptr %ptr
  %val4 = load volatile float, ptr %ptr
  %val5 = load volatile float, ptr %ptr
  %val6 = load volatile float, ptr %ptr
  %val7 = load volatile float, ptr %ptr
  %val8 = load volatile float, ptr %ptr
  %val9 = load volatile float, ptr %ptr
  %val10 = load volatile float, ptr %ptr
  %val11 = load volatile float, ptr %ptr
  %val12 = load volatile float, ptr %ptr
  %val13 = load volatile float, ptr %ptr
  %val14 = load volatile float, ptr %ptr
  %val15 = load volatile float, ptr %ptr
  %val16 = load volatile float, ptr %ptr

  %sqrt0 = call float @llvm.sqrt.f32(float %val0)
  %sqrt1 = call float @llvm.sqrt.f32(float %val1)
  %sqrt2 = call float @llvm.sqrt.f32(float %val2)
  %sqrt3 = call float @llvm.sqrt.f32(float %val3)
  %sqrt4 = call float @llvm.sqrt.f32(float %val4)
  %sqrt5 = call float @llvm.sqrt.f32(float %val5)
  %sqrt6 = call float @llvm.sqrt.f32(float %val6)
  %sqrt7 = call float @llvm.sqrt.f32(float %val7)
  %sqrt8 = call float @llvm.sqrt.f32(float %val8)
  %sqrt9 = call float @llvm.sqrt.f32(float %val9)
  %sqrt10 = call float @llvm.sqrt.f32(float %val10)
  %sqrt11 = call float @llvm.sqrt.f32(float %val11)
  %sqrt12 = call float @llvm.sqrt.f32(float %val12)
  %sqrt13 = call float @llvm.sqrt.f32(float %val13)
  %sqrt14 = call float @llvm.sqrt.f32(float %val14)
  %sqrt15 = call float @llvm.sqrt.f32(float %val15)
  %sqrt16 = call float @llvm.sqrt.f32(float %val16)

  store volatile float %val0, ptr %ptr
  store volatile float %val1, ptr %ptr
  store volatile float %val2, ptr %ptr
  store volatile float %val3, ptr %ptr
  store volatile float %val4, ptr %ptr
  store volatile float %val5, ptr %ptr
  store volatile float %val6, ptr %ptr
  store volatile float %val7, ptr %ptr
  store volatile float %val8, ptr %ptr
  store volatile float %val9, ptr %ptr
  store volatile float %val10, ptr %ptr
  store volatile float %val11, ptr %ptr
  store volatile float %val12, ptr %ptr
  store volatile float %val13, ptr %ptr
  store volatile float %val14, ptr %ptr
  store volatile float %val15, ptr %ptr
  store volatile float %val16, ptr %ptr

  store volatile float %sqrt0, ptr %ptr
  store volatile float %sqrt1, ptr %ptr
  store volatile float %sqrt2, ptr %ptr
  store volatile float %sqrt3, ptr %ptr
  store volatile float %sqrt4, ptr %ptr
  store volatile float %sqrt5, ptr %ptr
  store volatile float %sqrt6, ptr %ptr
  store volatile float %sqrt7, ptr %ptr
  store volatile float %sqrt8, ptr %ptr
  store volatile float %sqrt9, ptr %ptr
  store volatile float %sqrt10, ptr %ptr
  store volatile float %sqrt11, ptr %ptr
  store volatile float %sqrt12, ptr %ptr
  store volatile float %sqrt13, ptr %ptr
  store volatile float %sqrt14, ptr %ptr
  store volatile float %sqrt15, ptr %ptr
  store volatile float %sqrt16, ptr %ptr

  ret void
}

; Check that a call to the normal sqrtf function is lowered.
define float @f8(float %dummy, float %val) {
; CHECK-LABEL: f8:
; CHECK: sqebr %f0, %f2
; CHECK: cebr %f0, %f0
; CHECK: bnor %r14
; CHECK: {{ler|ldr}} %f0, %f2
; CHECK: jg sqrtf@PLT
  %res = tail call float @sqrtf(float %val)
  ret float %res
}
