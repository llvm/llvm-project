; Test 64-bit atomic XORs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check XORs of a variable.
define i64 @f1(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^ ]*]]:
; CHECK: lgr %r0, %r2
; CHECK: xgr %r0, %r4
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw xor ptr %src, i64 %b seq_cst
  ret i64 %res
}

; Check the lowest useful XILF value.
define i64 @f2(i64 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: lg %r2, 0(%r3)
; CHECK: [[LABEL:\.[^ ]*]]:
; CHECK: lgr %r0, %r2
; CHECK: xilf %r0, 1
; CHECK: csg %r2, %r0, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw xor ptr %src, i64 1 seq_cst
  ret i64 %res
}

; Check the high end of the XILF range.
define i64 @f3(i64 %dummy, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: xilf %r0, 4294967295
; CHECK: br %r14
  %res = atomicrmw xor ptr %src, i64 4294967295 seq_cst
  ret i64 %res
}

; Check the lowest useful XIHF value, which is one greater than above.
define i64 @f4(i64 %dummy, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: xihf %r0, 1
; CHECK: br %r14
  %res = atomicrmw xor ptr %src, i64 4294967296 seq_cst
  ret i64 %res
}

; Check the next value up, which must use a register.  (We could use
; combinations of XIH* and XIL* instead, but that isn't implemented.)
define i64 @f5(i64 %dummy, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: xgr
; CHECK: br %r14
  %res = atomicrmw xor ptr %src, i64 4294967297 seq_cst
  ret i64 %res
}

; Check the high end of the XIHF range.
define i64 @f6(i64 %dummy, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: xihf %r0, 4294967295
; CHECK: br %r14
  %res = atomicrmw xor ptr %src, i64 -4294967296 seq_cst
  ret i64 %res
}

; Check the next value up, which must use a register.
define i64 @f7(i64 %dummy, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: xgr
; CHECK: br %r14
  %res = atomicrmw xor ptr %src, i64 -4294967295 seq_cst
  ret i64 %res
}
