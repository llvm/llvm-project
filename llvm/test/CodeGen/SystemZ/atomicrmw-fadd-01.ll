; Test atomic float addition. Expect a compare-and-swap loop.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define float @f1(ptr %src, float %b) {
; CHECK-LABEL: f1:
; CHECK: le [[F:%f[0-9]+]], 0(%r2)
; CHECK: [[L:\.L.+]]:
; CHECK: ler [[COPY_F:%f[0-9]+]], [[F]]
; CHECK-NEXT: aebr [[F]], %f0
; CHECK-NEXT: lgdr [[RO:%r[0-9]+]], [[F]]
; CHECK: srlg [[RO]], [[RO]], 32
; CHECK: lgdr [[RI:%r[0-9]+]], [[COPY_F]]
; CHECK: srlg [[RI]], [[RI]], 32
; CHECK: cs [[RI]], [[RO]], 0(%r2)
; CHECK: sllg [[RO]], [[RI]], 32
; CHECK: ldgr [[F]], [[RO]]
; CHECK: jl [[L]]
; CHECK: ler %f0, [[F]]
; CHECK: br %r14
  %res = atomicrmw fadd ptr %src, float %b seq_cst
  ret float %res
}
