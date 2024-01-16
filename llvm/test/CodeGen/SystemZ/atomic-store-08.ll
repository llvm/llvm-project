; Test long double atomic stores. The atomic store is converted to i128 by
; the AtomicExpand pass.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

define void @f1(ptr %dst, ptr %src) {
; CHECK-LABEL: f1:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lg %r1, 8(%r3)
; CHECK-NEXT:    lg %r0, 0(%r3)
; CHECK-NEXT:    stpq %r0, 0(%r2)
; CHECK-NEXT:    bcr 1{{[45]}}, %r0
; CHECK-NEXT:    br %r14
  %val = load fp128, ptr %src, align 8
  store atomic fp128 %val, ptr %dst seq_cst, align 16
  ret void
}

define void @f2(ptr %dst, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: brasl %r14, __atomic_store@PLT
  %val = load fp128, ptr %src, align 8
  store atomic fp128 %val, ptr %dst seq_cst, align 8
  ret void
}
