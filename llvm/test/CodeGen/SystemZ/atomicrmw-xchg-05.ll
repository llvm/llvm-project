; Test float atomic exchange.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define float @f1(ptr %src, float %b) {
; CHECK-LABEL: f1:
; CHECK: l [[RI:%r[0-9]+]], 0(%r2)
; CHECK: lgdr [[RO:%r[0-9]+]], %f0
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: cs [[RI]], [[RO]], 0(%r2)
; CHECK: jl [[LABEL]]
; CHECK: sllg [[RI]], [[RI]], 32
; CHECK: ldgr %f0, [[RI]]
; CHECK: br %r14
  %res = atomicrmw xchg ptr %src, float %b seq_cst
  ret float %res
}
