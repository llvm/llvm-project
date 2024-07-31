; Test decrementing until to a minimum value. Expect a compare-and-swap loop.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define i64 @f1(ptr %src, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: lgr [[SRC:%r[0-9]+]], %r2
; CHECK: lg [[RI:%r[0-9]+]], 0(%r2)
; CHECK: j [[L2:\.L.+]]
; CHECK: [[L1:\.L.+]]:
; CHECK: csg [[RI]], [[RO:%r[0-9]+]], 0([[SRC]])
; CHECK: je [[L4:\.L.+]]
; CHECK: [[L2]]:
; CHECK: lgr [[RO]], [[RI]]
; CHECK: slgfi [[RO]], 1
; CHECK: lgr [[RB:%r[0-9]+]], %r3
; CHECK: clgrjh [[RI]], %r3, [[L3:\.L.+]]
; CHECK: lgr [[RB]], [[RO]]
; CHECK: [[L3]]:
; CHECK: lgr [[RO]], [[RI]]
; CHECK: slgfi [[RO]], 1
; CHECK: lgr [[RO]], %r3
; CHECK: jle [[L1]]
; CHECK: [[L4]]:
; CHECK: br %r14
  %res = atomicrmw udec_wrap ptr %src, i64 %b seq_cst
  ret i64 %res
}
