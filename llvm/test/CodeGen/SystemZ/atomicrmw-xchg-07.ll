; Test long double atomic exchange.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(ptr %ret, ptr %src, ptr %b) {
; CHECK-LABEL: f1:
; CHECK: lg [[RH:%r[0-9]+]], 8(%r4)
; CHECK: lgr [[RET:%r[0-9]+]], %r2
; CHECK: lg [[RL:%r[0-9]+]], 0(%r4)
; CHECK: stg [[RH]], 168(%r15)
; CHECK: la %r2, 176(%r15)
; CHECK: la %r4, 160(%r15)
; CHECK: stg [[RL]], 160(%r15)
; CHECK: brasl %r14, __sync_lock_test_and_set_16@PLT
; CHECK: lg [[RH2:%r[0-9]+]], 184(%r15)
; CHECK: lg [[RL2:%r[0-9]+]], 176(%r15)
; CHECK: stg [[RH]], 8([[RET]])
; CHECK: stg [[RL]], 0([[RET]])
; CHECK: br %r14
  %val = load fp128, ptr %b, align 8
  %res = atomicrmw xchg ptr %src, fp128 %val seq_cst
  store fp128 %res, ptr %ret, align 8
  ret void
}
