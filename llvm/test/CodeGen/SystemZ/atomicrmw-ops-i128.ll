; Test i128 atomicrmw operations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s

; Check register exchange.
define i128 @f1(i128 %dummy, ptr %src, i128 %b) {
; CHECK-LABEL: f1:
; CHECK: brasl %r14, __sync_lock_test_and_set_16@PLT
; CHECK: br %r14
  %res = atomicrmw xchg ptr %src, i128 %b seq_cst
  ret i128 %res
}

; Check addition of a variable.
define i128 @f2(i128 %dummy, ptr %src, i128 %b) {
; CHECK-LABEL: f2:
; CHECK: brasl %r14, __sync_fetch_and_add_16@PLT
; CHECK: br %r14
  %res = atomicrmw add ptr %src, i128 %b seq_cst
  ret i128 %res
}

; Check subtraction of a variable.
define i128 @f3(i128 %dummy, ptr %src, i128 %b) {
; CHECK-LABEL: f3:
; CHECK: brasl %r14, __sync_fetch_and_sub_16@PLT
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i128 %b seq_cst
  ret i128 %res
}

; Check AND of a variable.
define i128 @f4(i128 %dummy, ptr %src, i128 %b) {
; CHECK-LABEL: f4:
; CHECK: brasl %r14, __sync_fetch_and_and_16@PLT
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i128 %b seq_cst
  ret i128 %res
}

; Check NAND of a variable.
define i128 @f5(i128 %dummy, ptr %src, i128 %b) {
; CHECK-LABEL: f5:
; CHECK: brasl %r14, __sync_fetch_and_nand_16@PLT
; CHECK: br %r14
  %res = atomicrmw nand ptr %src, i128 %b seq_cst
  ret i128 %res
}

; Check OR of a variable.
define i128 @f6(i128 %dummy, ptr %src, i128 %b) {
; CHECK-LABEL: f6:
; CHECK: brasl %r14, __sync_fetch_and_or_16@PLT
; CHECK: br %r14
  %res = atomicrmw or ptr %src, i128 %b seq_cst
  ret i128 %res
}

; Check XOR of a variable.
define i128 @f7(i128 %dummy, ptr %src, i128 %b) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, __sync_fetch_and_xor_16@PLT
; CHECK: br %r14
  %res = atomicrmw xor ptr %src, i128 %b seq_cst
  ret i128 %res
}

; Check signed minimum.
define i128 @f8(i128 %dummy, ptr %src, i128 %b) {
; CHECK-LABEL: f8:
; CHECK: brasl %r14, __sync_fetch_and_min_16@PLT
; CHECK: br %r14
  %res = atomicrmw min ptr %src, i128 %b seq_cst
  ret i128 %res
}

; Check signed maximum.
define i128 @f9(i128 %dummy, ptr %src, i128 %b) {
; CHECK-LABEL: f9:
; CHECK: brasl %r14, __sync_fetch_and_max_16@PLT
; CHECK: br %r14
  %res = atomicrmw max ptr %src, i128 %b seq_cst
  ret i128 %res
}

; Check unsigned minimum.
define i128 @f10(i128 %dummy, ptr %src, i128 %b) {
; CHECK-LABEL: f10:
; CHECK: brasl %r14, __sync_fetch_and_umin_16@PLT
; CHECK: br %r14
  %res = atomicrmw umin ptr %src, i128 %b seq_cst
  ret i128 %res
}

; Check unsigned maximum.
define i128 @f11(i128 %dummy, ptr %src, i128 %b) {
; CHECK-LABEL: f11:
; CHECK: brasl %r14, __sync_fetch_and_umax_16@PLT
; CHECK: br %r14
  %res = atomicrmw umax ptr %src, i128 %b seq_cst
  ret i128 %res
}

