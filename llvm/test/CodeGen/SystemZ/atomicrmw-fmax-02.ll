; Test atomic double maximum.
; Expect a libcall in a compare-and-swap loop.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define double @f1(ptr %src, double %b) {
; CHECK-LABEL: f1:
; CHECK: lgr [[RB:%r[0-9]+]], %r2
; CHECK: ld [[FB:%f[0-9]+]], 0(%r2)
; CHECK: ldr [[FSRC:%f[0-9]+]], %f0
; CHECK: [[L:\.L.+]]:
; CHECK: ldr %f0, [[FB]]
; CHECK: ldr %f2, [[FSRC]]
; CHECK: brasl %r14, fmax@PLT
; CHECK: lgdr [[RO:%r[0-9]+]], %f0
; CHECK: lgdr [[RI:%r[0-9]+]], [[FB]]
; CHECK: csg [[RI]], [[RO]], 0([[RB]])
; CHECK: ldgr [[FB]], [[RI]]
; CHECK: jl [[L]]
; CHECK: ldr %f0, [[FB]]
; CHECK: br %r14
  %res = atomicrmw fmax ptr %src, double %b seq_cst
  ret double %res
}
