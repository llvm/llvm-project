; Test long double atomic exchange.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(ptr align 16 %ret, ptr align 16 %src, ptr align 16 %b) {
; CHECK-LABEL: f1:
; CHECK:       lg      %r1, 8(%r4)
; CHECK-NEXT:  lg      %r0, 0(%r4)
; CHECK-NEXT:  lg      %r4, 8(%r3)
; CHECK-NEXT:  lg      %r5, 0(%r3)
; CHECK-NEXT:.LBB0_1:                          # %atomicrmw.start
; CHECK-NEXT:                                  # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:  lgr     %r12, %r5
; CHECK-NEXT:  lgr     %r13, %r4
; CHECK-NEXT:  cdsg    %r12, %r0, 0(%r3)
; CHECK-NEXT:  lgr     %r4, %r13
; CHECK-NEXT:  lgr     %r5, %r12
; CHECK-NEXT:  jl      .LBB0_1
; CHECK-NEXT:# %bb.2:                          # %atomicrmw.end
; CHECK-NEXT:  stg     %r5, 0(%r2)
; CHECK-NEXT:  stg     %r4, 8(%r2)
; CHECK-NEXT:  lmg     %r12, %r15, 96(%r15)
; CHECK-NEXT:  br      %r14
  %val = load fp128, ptr %b, align 16
  %res = atomicrmw xchg ptr %src, fp128 %val seq_cst
  store fp128 %res, ptr %ret, align 16
  ret void
}
