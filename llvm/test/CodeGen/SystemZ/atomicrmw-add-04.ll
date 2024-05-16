; Test 64-bit atomic additions.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check addition of a variable.
define i64 @f1(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: lgr %r0, %r2
; CHECK: agr %r0, %r4
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw add ptr %src, i64 %b seq_cst
  ret i64 %res
}

; Check addition of 1.
define i64 @f2(i64 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: la %r0, 1(%r2)
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw add ptr %src, i64 1 seq_cst
  ret i64 %res
}

; Check use of LAY.
define i64 @f3(i64 %dummy, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: lay %r0, 32767(%r2)
; CHECK: br %r14
  %res = atomicrmw add ptr %src, i64 32767 seq_cst
  ret i64 %res
}

; Check the high end of the AGFI range.
define i64 @f4(i64 %dummy, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: agfi %r0, 2147483647
; CHECK: br %r14
  %res = atomicrmw add ptr %src, i64 2147483647 seq_cst
  ret i64 %res
}

; Check the next value up, which uses an ALGFI.
define i64 @f5(i64 %dummy, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: algfi %r0, 2147483648
; CHECK: br %r14
  %res = atomicrmw add ptr %src, i64 2147483648 seq_cst
  ret i64 %res
}

; Check addition of -1, which can use LAY.
define i64 @f6(i64 %dummy, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: lay %r0, -1(%r2)
; CHECK: br %r14
  %res = atomicrmw add ptr %src, i64 -1 seq_cst
  ret i64 %res
}

; LAY still OK.
define i64 @f7(i64 %dummy, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: lay %r0, -32769(%r2)
; CHECK: br %r14
  %res = atomicrmw add ptr %src, i64 -32769 seq_cst
  ret i64 %res
}

; Check the low end of the AGFI range.
define i64 @f8(i64 %dummy, ptr %src) {
; CHECK-LABEL: f8:
; CHECK: agfi %r0, -2147483648
; CHECK: br %r14
  %res = atomicrmw add ptr %src, i64 -2147483648 seq_cst
  ret i64 %res
}

; Check the next value down, which uses an SLGFI.
define i64 @f9(i64 %dummy, ptr %src) {
; CHECK-LABEL: f9:
; CHECK: slgfi %r0, 2147483649
; CHECK: br %r14
  %res = atomicrmw add ptr %src, i64 -2147483649 seq_cst
  ret i64 %res
}
