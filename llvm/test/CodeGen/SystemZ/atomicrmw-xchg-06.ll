; Test double atomic exchange.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define double @f1(ptr %src, double %b) {
; CHECK-LABEL: f1:
; CHECK: lg [[RI:%r[0-9]+]], 0(%r2)
; CHECK: lgdr [[RO:%r[0-9]+]], %f0
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: csg [[RI]], [[RO]], 0(%r2)
; CHECK: jl [[LABEL]]
; CHECK: ldgr %f0, [[RI]]
; CHECK: br %r14
  %res = atomicrmw xchg ptr %src, double %b seq_cst
  ret double %res
}
