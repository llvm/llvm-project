; Test atomic double addition. Expect a compare-and-swap loop.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define double @f1(ptr %src, double %b) {
; CHECK-LABEL: f1:
; CHECK: ld [[F:%f[0-9]+]], 0(%r2)
; CHECK: [[L:\.L.+]]:
; CHECK: lgdr [[RI:%r[0-9]+]], [[F]]
; CHECK: adbr [[F]], %f0
; CHECK: lgdr [[RO:%r[0-9]+]], [[F]]
; CHECK: csg [[RI]], [[RO]], 0(%r2)
; CHECK: ldgr [[F]], [[RI]]
; CHECK: jl [[L]]
; CHECK: ldr %f0, [[F]]
; CHECK: br %r14
  %res = atomicrmw fadd ptr %src, double %b seq_cst
  ret double %res
}
