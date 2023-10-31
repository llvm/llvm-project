; Test atomic long double subtraction. Expect a compare-and-swap loop.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(ptr %ret, ptr %src, ptr %b) {
; CHECK-LABEL: f1:
; CHECK: [[FBL:%f[0-9]+]], 0(%r4)
; CHECK: [[FBH:%f[0-9]+]], 8(%r4)
; CHECK: [[FSL:%f[0-9]+]], 0(%r3)
; CHECK: [[FSH:%f[0-9]+]], 8(%r3)
; CHECK: [[LABEL:\.L.+]]:
; CHECK: lgdr [[RISH:%r[0-9]+]], [[FSH]]
; CHECK: lgdr [[RISL:%r[0-9]+]], [[FSL]]
; CHECK: sxbr [[FSL]], [[FBL]]
; CHECK: lgdr [[ROSH:%r[0-9]+]], [[FSH]]
; CHECK: lgdr [[ROSL:%r[0-9]+]], [[FSL]]
; CHECK: cdsg [[RISL]], [[ROSL]], 0(%r3)
; CHECK: stg [[RISH]], 168(%r15)
; CHECK: stg [[RISL]], 160(%r15)
; CHECK: ld [[FSL]], 160(%r15)
; CHECK: ld [[FSH]], 168(%r15)
; CHECK: jl [[LABEL]]
; CHECK: std [[FSL]], 0(%r2)
; CHECK: std [[FSH]], 8(%r2)
; CHECK: br %r14
  %val = load fp128, ptr %b
  %res = atomicrmw fsub ptr %src, fp128 %val seq_cst
  store fp128 %res, ptr %ret
  ret void
}
