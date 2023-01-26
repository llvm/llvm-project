; Test 32-bit atomic ORs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check ORs of a variable.
define i32 @f1(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: l %r2, 0(%r3)
; CHECK: [[LABEL:\.[^ ]*]]:
; CHECK: lr %r0, %r2
; CHECK: or %r0, %r4
; CHECK: cs %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i32 %b seq_cst
  ret i32 %res
}

; Check the lowest useful OILL value.
define i32 @f2(i32 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: l %r2, 0(%r3)
; CHECK: [[LABEL:\.[^ ]*]]:
; CHECK: lr %r0, %r2
; CHECK: oill %r0, 1
; CHECK: cs %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i32 1 seq_cst
  ret i32 %res
}

; Check the high end of the OILL range.
define i32 @f3(i32 %dummy, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: oill %r0, 65535
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i32 65535 seq_cst
  ret i32 %res
}

; Check the lowest useful OILH value, which is the next value up.
define i32 @f4(i32 %dummy, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: oilh %r0, 1
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i32 65536 seq_cst
  ret i32 %res
}

; Check the lowest useful OILF value, which is the next value up.
define i32 @f5(i32 %dummy, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: oilf %r0, 65537
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i32 65537 seq_cst
  ret i32 %res
}

; Check the high end of the OILH range.
define i32 @f6(i32 %dummy, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: oilh %r0, 65535
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i32 -65536 seq_cst
  ret i32 %res
}

; Check the next value up, which must use OILF.
define i32 @f7(i32 %dummy, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: oilf %r0, 4294901761
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i32 -65535 seq_cst
  ret i32 %res
}

; Check the largest useful OILF value.
define i32 @f8(i32 %dummy, ptr %src) {
; CHECK-LABEL: f8:
; CHECK: oilf %r0, 4294967294
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i32 -2 seq_cst
  ret i32 %res
}
