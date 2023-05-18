; Test 64-bit floating-point loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test the low end of the LD range.
define double @f1(ptr %src) {
; CHECK-LABEL: f1:
; CHECK: ld %f0, 0(%r2)
; CHECK: br %r14
  %val = load double, ptr %src
  ret double %val
}

; Test the high end of the LD range.
define double @f2(ptr %src) {
; CHECK-LABEL: f2:
; CHECK: ld %f0, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 511
  %val = load double, ptr %ptr
  ret double %val
}

; Check the next doubleword up, which should use LDY instead of LD.
define double @f3(ptr %src) {
; CHECK-LABEL: f3:
; CHECK: ldy %f0, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 512
  %val = load double, ptr %ptr
  ret double %val
}

; Check the high end of the aligned LDY range.
define double @f4(ptr %src) {
; CHECK-LABEL: f4:
; CHECK: ldy %f0, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 65535
  %val = load double, ptr %ptr
  ret double %val
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f5(ptr %src) {
; CHECK-LABEL: f5:
; CHECK: agfi %r2, 524288
; CHECK: ld %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 65536
  %val = load double, ptr %ptr
  ret double %val
}

; Check the high end of the negative aligned LDY range.
define double @f6(ptr %src) {
; CHECK-LABEL: f6:
; CHECK: ldy %f0, -8(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 -1
  %val = load double, ptr %ptr
  ret double %val
}

; Check the low end of the LDY range.
define double @f7(ptr %src) {
; CHECK-LABEL: f7:
; CHECK: ldy %f0, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 -65536
  %val = load double, ptr %ptr
  ret double %val
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f8(ptr %src) {
; CHECK-LABEL: f8:
; CHECK: agfi %r2, -524296
; CHECK: ld %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 -65537
  %val = load double, ptr %ptr
  ret double %val
}

; Check that LD allows an index.
define double @f9(i64 %src, i64 %index) {
; CHECK-LABEL: f9:
; CHECK: ld %f0, 4095({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to ptr
  %val = load double, ptr %ptr
  ret double %val
}

; Check that LDY allows an index.
define double @f10(i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: ldy %f0, 4096({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to ptr
  %val = load double, ptr %ptr
  ret double %val
}
