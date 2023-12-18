; Test long double atomic stores. Expect a libcall.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(ptr %dst, ptr %src) {
; CHECK-LABEL: f1:
; CHECK: ld [[FL:%f[0-9]+]], 0(%r3)
; CHECK: ld [[FH:%f[0-9]+]], 8(%r3)
; CHECK: lgr %r3, %r2
; CHECK: std [[FL]], 160(%r15)
; CHECK: std [[FH]], 168(%r15)
; CHECK: la %r4, 160(%r15)
; CHECK: lghi %r2, 16
; CHECK: lhi %r5, 5
; CHECK: brasl %r14, __atomic_store@PLT
; CHECK: br %r14
  %val = load fp128, ptr %src, align 8
  store atomic fp128 %val, ptr %dst seq_cst, align 8
  ret void
}
