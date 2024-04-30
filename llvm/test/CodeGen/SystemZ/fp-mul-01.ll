; Test multiplication of two f32s, producing an f32 result.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @foo()

; Check register multiplication.
define float @f1(float %f1, float %f2) {
; CHECK-LABEL: f1:
; CHECK: meebr %f0, %f2
; CHECK: br %r14
  %res = fmul float %f1, %f2
  ret float %res
}

; Check the low end of the MEEB range.
define float @f2(float %f1, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: meeb %f0, 0(%r2)
; CHECK: br %r14
  %f2 = load float, ptr %ptr
  %res = fmul float %f1, %f2
  ret float %res
}

; Check the high end of the aligned MEEB range.
define float @f3(float %f1, ptr %base) {
; CHECK-LABEL: f3:
; CHECK: meeb %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 1023
  %f2 = load float, ptr %ptr
  %res = fmul float %f1, %f2
  ret float %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define float @f4(float %f1, ptr %base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: meeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 1024
  %f2 = load float, ptr %ptr
  %res = fmul float %f1, %f2
  ret float %res
}

; Check negative displacements, which also need separate address logic.
define float @f5(float %f1, ptr %base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -4
; CHECK: meeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, ptr %base, i64 -1
  %f2 = load float, ptr %ptr
  %res = fmul float %f1, %f2
  ret float %res
}

; Check that MEEB allows indices.
define float @f6(float %f1, ptr %base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 2
; CHECK: meeb %f0, 400(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float, ptr %base, i64 %index
  %ptr2 = getelementptr float, ptr %ptr1, i64 100
  %f2 = load float, ptr %ptr2
  %res = fmul float %f1, %f2
  ret float %res
}

; Check that multiplications of spilled values can use MEEB rather than MEEBR.
define float @f7(ptr %ptr0) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK-SCALAR: meeb %f0, 16{{[04]}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr float, ptr %ptr0, i64 2
  %ptr2 = getelementptr float, ptr %ptr0, i64 4
  %ptr3 = getelementptr float, ptr %ptr0, i64 6
  %ptr4 = getelementptr float, ptr %ptr0, i64 8
  %ptr5 = getelementptr float, ptr %ptr0, i64 10
  %ptr6 = getelementptr float, ptr %ptr0, i64 12
  %ptr7 = getelementptr float, ptr %ptr0, i64 14
  %ptr8 = getelementptr float, ptr %ptr0, i64 16
  %ptr9 = getelementptr float, ptr %ptr0, i64 18
  %ptr10 = getelementptr float, ptr %ptr0, i64 20

  %val0 = load float, ptr %ptr0
  %val1 = load float, ptr %ptr1
  %val2 = load float, ptr %ptr2
  %val3 = load float, ptr %ptr3
  %val4 = load float, ptr %ptr4
  %val5 = load float, ptr %ptr5
  %val6 = load float, ptr %ptr6
  %val7 = load float, ptr %ptr7
  %val8 = load float, ptr %ptr8
  %val9 = load float, ptr %ptr9
  %val10 = load float, ptr %ptr10

  %ret = call float @foo()

  %mul0 = fmul float %ret, %val0
  %mul1 = fmul float %mul0, %val1
  %mul2 = fmul float %mul1, %val2
  %mul3 = fmul float %mul2, %val3
  %mul4 = fmul float %mul3, %val4
  %mul5 = fmul float %mul4, %val5
  %mul6 = fmul float %mul5, %val6
  %mul7 = fmul float %mul6, %val7
  %mul8 = fmul float %mul7, %val8
  %mul9 = fmul float %mul8, %val9
  %mul10 = fmul float %mul9, %val10

  ret float %mul10
}

; Check that reassociation flags do not get in the way of MEEB.
define float @f8(ptr %x) {
; CHECK-LABEL: f8:
; CHECK: meeb %f0
entry:
  %0 = load float, ptr %x, align 8
  %arrayidx1 = getelementptr inbounds float, ptr %x, i64 1
  %1 = load float, ptr %arrayidx1, align 8
  %add = fmul reassoc nsz arcp contract afn float %1, %0
  ret float %add
}
