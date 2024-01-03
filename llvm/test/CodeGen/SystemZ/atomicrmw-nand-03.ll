; Test 32-bit atomic NANDs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check NANDs of a variable.
define i32 @f1(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: l %r2, 0(%r3)
; CHECK: [[LABEL:\.[^ ]*]]:
; CHECK: lr %r0, %r2
; CHECK: nr %r0, %r4
; CHECK: xilf %r0, 4294967295
; CHECK: cs %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i32 %b seq_cst
  ret i32 %res
}

; Check NANDs with different constant operands.
define i32 @f2(i32 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: l %r2, 0(%r3)
; CHECK: [[LABEL:\.[^ ]*]]:
; CHECK: lr %r0, %r2
; CHECK: xilf %r0, 4294967295
; CHECK: oilf %r0, 4294967294
; CHECK: cs %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i32 1 seq_cst
  ret i32 %res
}

define i32 @f3(i32 %dummy, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: xilf %r0, 4294967295
; CHECK: oilh %r0, 65535
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i32 65535 seq_cst
  ret i32 %res
}

define i32 @f4(i32 %dummy, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: xilf %r0, 4294967295
; CHECK: oilf %r0, 4294901759
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i32 65536 seq_cst
  ret i32 %res
}

define i32 @f5(i32 %dummy, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: xilf %r0, 4294967295
; CHECK: oill %r0, 1
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i32 -2 seq_cst
  ret i32 %res
}

define i32 @f6(i32 %dummy, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: xilf %r0, 4294967295
; CHECK: oill %r0, 65535
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i32 -65536 seq_cst
  ret i32 %res
}

define i32 @f7(i32 %dummy, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: xilf %r0, 4294967295
; CHECK: oilh %r0, 1
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i32 -65537 seq_cst
  ret i32 %res
}

define i32 @f8(i32 %dummy, ptr %src) {
; CHECK-LABEL: f8:
; CHECK: xilf %r0, 4294967295
; CHECK: oilf %r0, 65537
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i32 -65538 seq_cst
  ret i32 %res
}
