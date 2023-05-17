; Test 32-bit atomic ANDs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check ANDs of a variable.
define i32 @f1(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: l %r2, 0(%r3)
; CHECK: [[LABEL:\.[^ ]*]]:
; CHECK: lr %r0, %r2
; CHECK: nr %r0, %r4
; CHECK: cs %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i32 %b seq_cst
  ret i32 %res
}

; Check ANDs of 1.
define i32 @f2(i32 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: l %r2, 0(%r3)
; CHECK: [[LABEL:\.[^ ]*]]:
; CHECK: lr %r0, %r2
; CHECK: nilf %r0, 1
; CHECK: cs %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i32 1 seq_cst
  ret i32 %res
}

; Check ANDs of the low end of the NILH range.
define i32 @f3(i32 %dummy, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: nilh %r0, 0
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i32 65535 seq_cst
  ret i32 %res
}

; Check the next value up, which must use NILF.
define i32 @f4(i32 %dummy, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: nilf %r0, 65536
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i32 65536 seq_cst
  ret i32 %res
}

; Check the largest useful NILL value.
define i32 @f5(i32 %dummy, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: nill %r0, 65534
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i32 -2 seq_cst
  ret i32 %res
}

; Check the low end of the NILL range.
define i32 @f6(i32 %dummy, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: nill %r0, 0
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i32 -65536 seq_cst
  ret i32 %res
}

; Check the largest useful NILH value, which is one less than the above.
define i32 @f7(i32 %dummy, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: nilh %r0, 65534
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i32 -65537 seq_cst
  ret i32 %res
}

; Check the highest useful NILF value, which is one less than the above.
define i32 @f8(i32 %dummy, ptr %src) {
; CHECK-LABEL: f8:
; CHECK: nilf %r0, 4294901758
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i32 -65538 seq_cst
  ret i32 %res
}
