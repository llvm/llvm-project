; Test long double atomic loads. These are emitted by the Clang FE as i128
; loads with a bitcast, and this test case gets converted into that form as
; well by the AtomicExpand pass.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

define void @f1(ptr %ret, ptr %src) {
; CHECK-LABEL: f1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lpq %r0, 0(%r3)
; CHECK-NEXT:    stg %r1, 8(%r2)
; CHECK-NEXT:    stg %r0, 0(%r2)
; CHECK-NEXT:    br %r14
  %val = load atomic fp128, ptr %src seq_cst, align 16
  store fp128 %val, ptr %ret, align 8
  ret void
}

define void @f2(ptr %ret, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: brasl %r14, __atomic_load@PLT
  %val = load atomic fp128, ptr %src seq_cst, align 8
  store fp128 %val, ptr %ret, align 8
  ret void
}
