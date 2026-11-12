; Test 64-bit floating-point addition.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -verify-machineinstrs | FileCheck %s
declare double @foo()

; Check register addition.
define double @f1(double %f1, double %f2) {
; CHECK-LABEL: f1:
; CHECK: adbr %f0, %f2
; CHECK: br %r14
  %res = fadd double %f1, %f2
  ret double %res
}

; Check the low end of the ADB range.
define double @f2(double %f1, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: adb %f0, 0(%r2)
; CHECK: br %r14
  %f2 = load double, ptr %ptr
  %res = fadd double %f1, %f2
  ret double %res
}

; Check the high end of the aligned ADB range.
define double @f3(double %f1, ptr %base) {
; CHECK-LABEL: f3:
; CHECK: adb %f0, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 511
  %f2 = load double, ptr %ptr
  %res = fadd double %f1, %f2
  ret double %res
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(double %f1, ptr %base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: adb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 512
  %f2 = load double, ptr %ptr
  %res = fadd double %f1, %f2
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(double %f1, ptr %base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -8
; CHECK: adb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 -1
  %f2 = load double, ptr %ptr
  %res = fadd double %f1, %f2
  ret double %res
}

; Check that ADB allows indices.
define double @f6(double %f1, ptr %base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 3
; CHECK: adb %f0, 800(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr double, ptr %base, i64 %index
  %ptr2 = getelementptr double, ptr %ptr1, i64 100
  %f2 = load double, ptr %ptr2
  %res = fadd double %f1, %f2
  ret double %res
}

; Check that additions of spilled values can use ADB rather than ADBR.
define double @f7(ptr %ptr0) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK-SCALAR: adb %f0, 160(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr double, ptr %ptr0, i64 2
  %ptr2 = getelementptr double, ptr %ptr0, i64 4
  %ptr3 = getelementptr double, ptr %ptr0, i64 6
  %ptr4 = getelementptr double, ptr %ptr0, i64 8
  %ptr5 = getelementptr double, ptr %ptr0, i64 10
  %ptr6 = getelementptr double, ptr %ptr0, i64 12
  %ptr7 = getelementptr double, ptr %ptr0, i64 14
  %ptr8 = getelementptr double, ptr %ptr0, i64 16
  %ptr9 = getelementptr double, ptr %ptr0, i64 18
  %ptr10 = getelementptr double, ptr %ptr0, i64 20

  %val0 = load double, ptr %ptr0
  %val1 = load double, ptr %ptr1
  %val2 = load double, ptr %ptr2
  %val3 = load double, ptr %ptr3
  %val4 = load double, ptr %ptr4
  %val5 = load double, ptr %ptr5
  %val6 = load double, ptr %ptr6
  %val7 = load double, ptr %ptr7
  %val8 = load double, ptr %ptr8
  %val9 = load double, ptr %ptr9
  %val10 = load double, ptr %ptr10

  %ret = call double @foo()

  %add0 = fadd double %ret, %val0
  %add1 = fadd double %add0, %val1
  %add2 = fadd double %add1, %val2
  %add3 = fadd double %add2, %val3
  %add4 = fadd double %add3, %val4
  %add5 = fadd double %add4, %val5
  %add6 = fadd double %add5, %val6
  %add7 = fadd double %add6, %val7
  %add8 = fadd double %add7, %val8
  %add9 = fadd double %add8, %val9
  %add10 = fadd double %add9, %val10

  ret double %add10
}

; Check that reassociation flags do not get in the way of ADB.
define double @f8(ptr %x) {
; CHECK-LABEL: f8:
; CHECK: ld %f0
; CHECK: adb %f0
; CHECK: br %r14
entry:
  %0 = load double, ptr %x, align 8
  %arrayidx1 = getelementptr inbounds double, ptr %x, i64 1
  %1 = load double, ptr %arrayidx1, align 8
  %add = fadd reassoc nsz arcp contract afn double %1, %0
  ret double %add
}
