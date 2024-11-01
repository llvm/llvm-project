; Test atomic float subtraction. Expect a compare-and-swap loop.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define float @f1(ptr %src, float %b) {
; CHECK-LABEL: f1:
; CHECK: le [[F:%f[0-9]+]], 0(%r2)
; CHECK: [[L:\.L.+]]:
; CHECK: lgdr [[RI:%r[0-9]+]], [[F]]
; CHECK: sebr [[F]], %f0
; CHECK: lgdr [[RO:%r[0-9]+]], [[F]]
; CHECK: srlg [[RO]], [[RO]], 32
; CHECK: srlg [[RI]], [[RI]], 32
; CHECK: cs [[RI]], [[RO]], 0(%r2)
; CHECK: sllg [[RI]], [[RI]], 32
; CHECK: ldgr [[F]], [[RI]]
; CHECK: jl [[L]]
; CHECK: ler %f0, [[F]]
; CHECK: br %r14
  %res = atomicrmw fsub ptr %src, float %b seq_cst
  ret float %res
}
