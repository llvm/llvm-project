; Test atomic long double minimum.
; Expect a libcall in a compare-and-swap loop.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(ptr %ret, ptr %src, ptr %b) {
; CHECK-LABEL: f1:
; CHECK: lgr [[SRC:%r[0-9]+]], %r3
; CHECK: ld [[FBL:%f[0-9]+]], 0(%r4)
; CHECK: ld [[FBH:%f[0-9]+]], 8(%r4)
; CHECK: ld [[FSL:%f[0-9]+]], 0(%r3)
; CHECK: ld [[FSH:%f[0-9]+]], 8(%r3)
; CHECK: lgr [[RET:%r[0-9]+]], %r2
; CHECK: [[L:\.L.+]]:
; CHECK: std [[FBL]], 160(%r15)
; CHECK: std [[FBH]], 168(%r15)
; CHECK: la %r2, 192(%r15)
; CHECK: la %r3, 176(%r15)
; CHECK: la %r4, 160(%r15)
; CHECK: std [[FSL]], 176(%r15)
; CHECK: std [[FSH]], 184(%r15)
; CHECK: brasl %r14, fminl@PLT
; CHECK: ld [[FL:%f[0-9]+]], 192(%r15)
; CHECK: ld [[FH:%f[0-9]+]], 200(%r15)
; CHECK: lgdr [[RH:%r[0-9]+]], [[FH]]
; CHECK: lgdr [[RL:%r[0-9]+]], [[FL]]
; CHECK: lgdr [[RSH:%r[0-9]+]], [[FSH]]
; CHECK: lgdr [[RSL:%r[0-9]+]], [[FSL]]
; CHECK: cdsg [[RSL]], [[RL]], 0([[SRC]])
; CHECK: stg [[RSH]], 216(%r15)
; CHECK: stg [[RSL]], 208(%r15)
; CHECK: ld [[FSL]], 208(%r15)
; CHECK: ld [[FSH]], 216(%r15)
; CHECK: jl [[L]]
; CHECK: std [[FSL]], 0([[RET]])
; CHECK: std [[FSH]], 8([[RET]])
; CHECK: br %r14
  %val = load fp128, ptr %b
  %res = atomicrmw fmin ptr %src, fp128 %val seq_cst
  store fp128 %res, ptr %ret
  ret void
}
