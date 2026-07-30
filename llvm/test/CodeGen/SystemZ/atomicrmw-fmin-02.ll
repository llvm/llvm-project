; Test atomic double minimum.
; Expect a libcall in a compare-and-swap loop.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define double @f1(ptr %src, double %b) {
; CHECK-LABEL: f1:
; CHECK: lgr [[SRC:%r[0-9]+]], %r2
; CHECK: ld [[FSRC:%f[0-9]+]], 0(%r2)
; CHECK: ldr [[FB:%f[0-9]+]], %f0
; CHECK: [[L:\.L.+]]:
; CHECK: ldr %f0, [[FSRC]]
; CHECK: ldr %f2, [[FB]]
; CHECK: brasl %r14, fmin@PLT
; CHECK: lgdr [[RO:%r[0-9]+]], %f0
; CHECK: lgdr [[RI:%r[0-9]+]], [[FSRC]]
; CHECK: csg [[RI]], [[RO]], 0([[SRC]])
; CHECK: ldgr [[FSRC]], [[RI]]
; CHECK: jl [[L]]
; CHECK: ldr %f0, [[FSRC]]
; CHECK: br %r14
  %res = atomicrmw fmin ptr %src, double %b seq_cst
  ret double %res
}
