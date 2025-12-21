; Test 64-bit atomic NANDs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check NANDs of a variable.
define i64 @f1(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: lgr %r0, %r2
; CHECK: ngr %r0, %r4
; CHECK: lcgr %r0, %r0
; CHECK: aghi %r0, -1
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 %b seq_cst
  ret i64 %res
}

; Check NANDs of 1, which are done using a register.
define i64 @f2(i64 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oihf %r0, 4294967295
; CHECK: oilf %r0, 4294967294
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 1 seq_cst
  ret i64 %res
}

define i64 @f3(i64 %dummy, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oihf %r0, 4294967294
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 8589934591 seq_cst
  ret i64 %res
}

define i64 @f4(i64 %dummy, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oihf %r0, 4294967293
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 12884901887 seq_cst
  ret i64 %res
}

define i64 @f5(i64 %dummy, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oihf %r0, 4294967292
; CHECK: oilf %r0, 4294967295
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 12884901888 seq_cst
  ret i64 %res
}

define i64 @f6(i64 %dummy, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oihh %r0, 65533
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 844424930131967 seq_cst
  ret i64 %res
}

define i64 @f7(i64 %dummy, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oihf %r0, 4294901759
; CHECK: oilf %r0, 4294967295
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 281474976710656 seq_cst
  ret i64 %res
}

define i64 @f8(i64 %dummy, ptr %src) {
; CHECK-LABEL: f8:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oill %r0, 5
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 -6 seq_cst
  ret i64 %res
}

define i64 @f9(i64 %dummy, ptr %src) {
; CHECK-LABEL: f9:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oill %r0, 65533
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 -65534 seq_cst
  ret i64 %res
}

define i64 @f10(i64 %dummy, ptr %src) {
; CHECK-LABEL: f10:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oilf %r0, 65537
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 -65538 seq_cst
  ret i64 %res
}

define i64 @f11(i64 %dummy, ptr %src) {
; CHECK-LABEL: f11:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oilh %r0, 5
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 -327681 seq_cst
  ret i64 %res
}

define i64 @f12(i64 %dummy, ptr %src) {
; CHECK-LABEL: f12:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oilh %r0, 65533
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 -4294770689 seq_cst
  ret i64 %res
}

define i64 @f13(i64 %dummy, ptr %src) {
; CHECK-LABEL: f13:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oilf %r0, 4294967293
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 -4294967294 seq_cst
  ret i64 %res
}

define i64 @f14(i64 %dummy, ptr %src) {
; CHECK-LABEL: f14:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oihl %r0, 5
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 -21474836481 seq_cst
  ret i64 %res
}

define i64 @f15(i64 %dummy, ptr %src) {
; CHECK-LABEL: f15:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oihl %r0, 65533
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 -281462091808769 seq_cst
  ret i64 %res
}

define i64 @f16(i64 %dummy, ptr %src) {
; CHECK-LABEL: f16:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oihh %r0, 5
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 -1407374883553281 seq_cst
  ret i64 %res
}

define i64 @f17(i64 %dummy, ptr %src) {
; CHECK-LABEL: f17:
; CHECK: lcgr %r0, %r2
; CHECK: aghi %r0, -1
; CHECK: oihf %r0, 65537
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i64 -281479271677953 seq_cst
  ret i64 %res
}
