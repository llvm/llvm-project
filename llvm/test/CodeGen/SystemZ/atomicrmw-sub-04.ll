; Test 64-bit atomic subtractions.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check subtraction of a variable.
define i64 @f1(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: lgr %r0, %r2
; CHECK: sgr %r0, %r4
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i64 %b seq_cst
  ret i64 %res
}

; Check subtraction of 1, which can use AGHI.
define i64 @f2(i64 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: lgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i64 1 seq_cst
  ret i64 %res
}

; Check the low end of the AGHI range.
define i64 @f3(i64 %dummy, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: aghi %r0, -32768
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i64 32768 seq_cst
  ret i64 %res
}

; Check the next value up, which must use AGFI.
define i64 @f4(i64 %dummy, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: agfi %r0, -32769
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i64 32769 seq_cst
  ret i64 %res
}

; Check the low end of the AGFI range.
define i64 @f5(i64 %dummy, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: agfi %r0, -2147483648
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i64 2147483648 seq_cst
  ret i64 %res
}

; Check the next value up, which must use a register operation.
define i64 @f6(i64 %dummy, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: sgr
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i64 2147483649 seq_cst
  ret i64 %res
}

; Check subtraction of -1, which can use AGHI.
define i64 @f7(i64 %dummy, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: aghi %r0, 1
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i64 -1 seq_cst
  ret i64 %res
}

; Check the high end of the AGHI range.
define i64 @f8(i64 %dummy, ptr %src) {
; CHECK-LABEL: f8:
; CHECK: aghi %r0, 32767
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i64 -32767 seq_cst
  ret i64 %res
}

; Check the next value down, which must use AGFI instead.
define i64 @f9(i64 %dummy, ptr %src) {
; CHECK-LABEL: f9:
; CHECK: agfi %r0, 32768
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i64 -32768 seq_cst
  ret i64 %res
}

; Check the high end of the AGFI range.
define i64 @f10(i64 %dummy, ptr %src) {
; CHECK-LABEL: f10:
; CHECK: agfi %r0, 2147483647
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i64 -2147483647 seq_cst
  ret i64 %res
}

; Check the next value down, which must use a register operation.
define i64 @f11(i64 %dummy, ptr %src) {
; CHECK-LABEL: f11:
; CHECK: sgr
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i64 -2147483648 seq_cst
  ret i64 %res
}
