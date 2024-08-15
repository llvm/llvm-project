; Test incrementing up to a maximum value. Expect a compare-and-swap loop.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define i64 @f1(ptr %src, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: lgr [[SRC:%r[0-9]+]], %r2
; CHECK: lg [[RI:%r[0-9]+]], 0(%r2)
; CHECK: j [[L2:\.L.+]]
; CHECK: [[L1:\.L.+]]:
; CHECK: csg [[RI]], [[RO:%r[0-9]+]], 0([[SRC]])
; CHECK: je [[L3:\.L.+]]
; CHECK: [[L2]]:
; CHECK: lghi [[RO]], 0
; CHECK: clgrjhe [[RI]], %r3, [[L1]]
; CHECK: la [[RO]], 1([[RI]])
; CHECK: j [[L1]]
; CHECK: [[L3]]:
; CHECK: br %r14
  %res = atomicrmw uinc_wrap ptr %src, i64 %b seq_cst
  ret i64 %res
}
