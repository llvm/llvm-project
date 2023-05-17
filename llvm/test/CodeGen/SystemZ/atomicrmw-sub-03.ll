; Test 32-bit atomic subtractions.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check subtraction of a variable.
define i32 @f1(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: l %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: lr %r0, %r2
; CHECK: sr %r0, %r4
; CHECK: cs %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i32 %b seq_cst
  ret i32 %res
}

; Check subtraction of 1, which can use AHI.
define i32 @f2(i32 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: l %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: lr %r0, %r2
; CHECK: ahi %r0, -1
; CHECK: cs %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i32 1 seq_cst
  ret i32 %res
}

; Check the low end of the AHI range.
define i32 @f3(i32 %dummy, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: ahi %r0, -32768
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i32 32768 seq_cst
  ret i32 %res
}

; Check the next value down, which must use AFI.
define i32 @f4(i32 %dummy, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: afi %r0, -32769
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i32 32769 seq_cst
  ret i32 %res
}

; Check the low end of the AFI range.
define i32 @f5(i32 %dummy, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: afi %r0, -2147483648
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i32 2147483648 seq_cst
  ret i32 %res
}

; Check the next value up, which gets treated as a positive operand.
define i32 @f6(i32 %dummy, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: afi %r0, 2147483647
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i32 2147483649 seq_cst
  ret i32 %res
}

; Check subtraction of -1, which can use AHI.
define i32 @f7(i32 %dummy, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: ahi %r0, 1
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i32 -1 seq_cst
  ret i32 %res
}

; Check the high end of the AHI range.
define i32 @f8(i32 %dummy, ptr %src) {
; CHECK-LABEL: f8:
; CHECK: ahi %r0, 32767
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i32 -32767 seq_cst
  ret i32 %res
}

; Check the next value down, which must use AFI instead.
define i32 @f9(i32 %dummy, ptr %src) {
; CHECK-LABEL: f9:
; CHECK: afi %r0, 32768
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i32 -32768 seq_cst
  ret i32 %res
}
