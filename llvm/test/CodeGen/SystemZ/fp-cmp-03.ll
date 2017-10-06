; Test 128-bit floating-point comparison.  The tests assume a z10 implementation
; of select, using conditional branches rather than LOCGR.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; There is no memory form of 128-bit comparison.
define i64 @f1(i64 %a, i64 %b, fp128 *%ptr, float %f2) {
; CHECK-LABEL: f1:
; CHECK-DAG: lxebr %f0, %f0
; CHECK-DAG: ld %f1, 0(%r4)
; CHECK-DAG: ld %f3, 8(%r4)
; CHECK: cxbr %f1, %f0
; CHECK-NEXT: ber %r14
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %f2x = fpext float %f2 to fp128
  %f1 = load fp128 , fp128 *%ptr
  %cond = fcmp oeq fp128 %f1, %f2x
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check comparison with zero.
define i64 @f2(i64 %a, i64 %b, fp128 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: ld %f0, 0(%r4)
; CHECK: ld %f2, 8(%r4)
; CHECK: ltxbr %f0, %f0
; CHECK-NEXT: ber %r14
; CHECK: lgr %r2, %r3
; CHECK: br %r14
  %f = load fp128 , fp128 *%ptr
  %cond = fcmp oeq fp128 %f, 0xL00000000000000000000000000000000
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}
