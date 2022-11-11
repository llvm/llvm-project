; Test 128-bit atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define i128 @f1(ptr %src) {
; CHECK-LABEL: f1:
; CHECK: lpq %r0, 0(%r3)
; CHECK-DAG: stg %r1, 8(%r2)
; CHECK-DAG: stg %r0, 0(%r2)
; CHECK: br %r14
  %val = load atomic i128, ptr %src seq_cst, align 16
  ret i128 %val
}
