; Test 128-bit integer atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

define i128 @f1(ptr %src) {
; CHECK-LABEL: f1:
; CHECK: # %bb.0:
; CHECK-NEXT: lpq %r0, 0(%r3)
; CHECK-NEXT: stg %r1, 8(%r2)
; CHECK-NEXT: stg %r0, 0(%r2)
; CHECK-NEXT: br %r14
  %val = load atomic i128, ptr %src seq_cst, align 16
  ret i128 %val
}

define i128 @f2(ptr %src) {
; CHECK-LABEL: f2:
; CHECK: brasl %r14, __atomic_load@PLT
  %val = load atomic i128, ptr %src seq_cst, align 8
  ret i128 %val
}
