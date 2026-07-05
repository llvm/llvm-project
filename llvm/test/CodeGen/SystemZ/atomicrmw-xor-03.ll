; Test 32-bit atomic XORs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check XORs of a variable.
define i32 @f1(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: l %r2, 0(%r3)
; CHECK: [[LABEL:\.[^ ]*]]:
; CHECK: lr %r0, %r2
; CHECK: xr %r0, %r4
; CHECK: cs %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw xor ptr %src, i32 %b seq_cst
  ret i32 %res
}

; Check the lowest useful constant.
define i32 @f2(i32 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: l %r2, 0(%r3)
; CHECK: [[LABEL:\.[^ ]*]]:
; CHECK: lr %r0, %r2
; CHECK: xilf %r0, 1
; CHECK: cs %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw xor ptr %src, i32 1 seq_cst
  ret i32 %res
}

; Check an arbitrary constant.
define i32 @f3(i32 %dummy, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: xilf %r0, 3000000000
; CHECK: br %r14
  %res = atomicrmw xor ptr %src, i32 3000000000 seq_cst
  ret i32 %res
}

; Check bitwise negation.
define i32 @f4(i32 %dummy, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: xilf %r0, 4294967295
; CHECK: br %r14
  %res = atomicrmw xor ptr %src, i32 -1 seq_cst
  ret i32 %res
}
