; Test 64-bit atomic ORs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check ORs of a variable.
define i64 @f1(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^ ]*]]:
; CHECK: lgr %r0, %r2
; CHECK: ogr %r0, %r4
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 %b seq_cst
  ret i64 %res
}

; Check the lowest useful OILL value.
define i64 @f2(i64 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^ ]*]]:
; CHECK: lgr %r0, %r2
; CHECK: oill %r0, 1
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 1 seq_cst
  ret i64 %res
}

; Check the high end of the OILL range.
define i64 @f3(i64 %dummy, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: oill %r0, 65535
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 65535 seq_cst
  ret i64 %res
}

; Check the lowest useful OILH value, which is the next value up.
define i64 @f4(i64 %dummy, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: oilh %r0, 1
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 65536 seq_cst
  ret i64 %res
}

; Check the lowest useful OILF value, which is the next value up again.
define i64 @f5(i64 %dummy, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: oilf %r0, 65537
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 65537 seq_cst
  ret i64 %res
}

; Check the high end of the OILH range.
define i64 @f6(i64 %dummy, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: oilh %r0, 65535
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 4294901760 seq_cst
  ret i64 %res
}

; Check the next value up, which must use OILF.
define i64 @f7(i64 %dummy, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: oilf %r0, 4294901761
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 4294901761 seq_cst
  ret i64 %res
}

; Check the high end of the OILF range.
define i64 @f8(i64 %dummy, ptr %src) {
; CHECK-LABEL: f8:
; CHECK: oilf %r0, 4294967295
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 4294967295 seq_cst
  ret i64 %res
}

; Check the lowest useful OIHL value, which is one greater than above.
define i64 @f9(i64 %dummy, ptr %src) {
; CHECK-LABEL: f9:
; CHECK: oihl %r0, 1
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 4294967296 seq_cst
  ret i64 %res
}

; Check the next value up, which must use a register.
define i64 @f10(i64 %dummy, ptr %src) {
; CHECK-LABEL: f10:
; CHECK: oihl %r0, 1
; CHECK: oill %r0, 1
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 4294967297 seq_cst
  ret i64 %res
}

; Check the high end of the OIHL range.
define i64 @f11(i64 %dummy, ptr %src) {
; CHECK-LABEL: f11:
; CHECK: oihl %r0, 65535
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 281470681743360 seq_cst
  ret i64 %res
}

; Check the lowest useful OIHH value, which is 1<<32 greater than above.
define i64 @f12(i64 %dummy, ptr %src) {
; CHECK-LABEL: f12:
; CHECK: oihh %r0, 1
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 281474976710656 seq_cst
  ret i64 %res
}

; Check the lowest useful OIHF value, which is 1<<32 greater again.
define i64 @f13(i64 %dummy, ptr %src) {
; CHECK-LABEL: f13:
; CHECK: oihf %r0, 65537
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 281479271677952 seq_cst
  ret i64 %res
}

; Check the high end of the OIHH range.
define i64 @f14(i64 %dummy, ptr %src) {
; CHECK-LABEL: f14:
; CHECK: oihh %r0, 65535
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 18446462598732840960 seq_cst
  ret i64 %res
}

; Check the next value up.
define i64 @f15(i64 %dummy, ptr %src) {
; CHECK-LABEL: f15:
; CHECK: oihh %r0, 65535
; CHECK: oill %r0, 1
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 18446462598732840961 seq_cst
  ret i64 %res
}

; Check the high end of the OIHF range.
define i64 @f16(i64 %dummy, ptr %src) {
; CHECK-LABEL: f16:
; CHECK: oihf %r0, 4294967295
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i64 -4294967296 seq_cst
  ret i64 %res
}
