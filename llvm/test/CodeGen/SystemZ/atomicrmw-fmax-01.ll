; Test atomic float maximum.
; Expect a libcall in a compare-and-swap loop.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define float @f1(ptr %src, float %b) {
; CHECK-LABEL: f1:
; CHECK: lgr [[SRC:%r[0-9]+]], %r2
; CHECK: le [[FSRC:%f[0-9]+]], 0(%r2)
; CHECK: ler [[FB:%f[0-9]+]], %f0
; CHECK: [[L:\.L.+]]:
; CHECK: ler %f0, [[FSRC]]
; CHECK: ler %f2, [[FB]]
; CHECK: brasl %r14, fmaxf@PLT
; CHECK: lgdr [[RO:%r[0-9]+]], %f0
; CHECK: srlg [[RO]], [[RO]], 32
; CHECK: lgdr [[RI:%r[0-9]+]], [[FSRC]]
; CHECK: srlg [[RI]], [[RI]], 32
; CHECK: cs [[RI]], [[RO]], 0([[SRC]])
; CHECK: sllg [[RO]], [[RI]], 32
; CHECK: ldgr [[FSRC]], [[RO]]
; CHECK: jl [[L]]
; CHECK: ler %f0, [[FSRC]]
; CHECK: br %r14
  %res = atomicrmw fmax ptr %src, float %b seq_cst
  ret float %res
}
