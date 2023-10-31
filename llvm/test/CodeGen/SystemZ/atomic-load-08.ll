; Test long double atomic loads. Expect a libcall.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(ptr %ret, ptr %src) {
; CHECK-LABEL: f1:
; CHECK: lgr [[RET:%r[0-9]+]], %r2
; CHECK: la %r4, 160(%r15)
; CHECK: lghi %r2, 16
; CHECK: lhi %r5, 5
; CHECK: brasl %r14, __atomic_load@PLT
; CHECK: ld [[FL:%f[0-9]+]], 160(%r15)
; CHECK: ld [[FH:%f[0-9]+]], 168(%r15)
; CHECK: std [[FL]], 0([[RET]])
; CHECK: std [[FH]], 8([[RET]])
; CHECK: br %r14
  %val = load atomic fp128, ptr %src seq_cst, align 8
  store fp128 %val, ptr %ret, align 8
  ret void
}
