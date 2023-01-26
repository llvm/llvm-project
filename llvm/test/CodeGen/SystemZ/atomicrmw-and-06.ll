; Test 64-bit atomic ANDs, z196 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check AND of a variable.
define i64 @f1(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: lang %r2, %r4, 0(%r3)
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 %b seq_cst
  ret i64 %res
}

; Check AND of -2, which needs a temporary.
define i64 @f2(i64 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: lghi [[TMP:%r[0-5]]], -2
; CHECK: lang %r2, [[TMP]], 0(%r3)
; CHECK: br %r14
  %res = atomicrmw and ptr %src, i64 -2 seq_cst
  ret i64 %res
}

; Check the high end of the LANG range.
define i64 @f3(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f3:
; CHECK: lang %r2, %r4, 524280(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 65535
  %res = atomicrmw and ptr %ptr, i64 %b seq_cst
  ret i64 %res
}

; Check the next doubleword up, which needs separate address logic.
define i64 @f4(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: agfi %r3, 524288
; CHECK: lang %r2, %r4, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 65536
  %res = atomicrmw and ptr %ptr, i64 %b seq_cst
  ret i64 %res
}

; Check the low end of the LANG range.
define i64 @f5(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f5:
; CHECK: lang %r2, %r4, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -65536
  %res = atomicrmw and ptr %ptr, i64 %b seq_cst
  ret i64 %res
}

; Check the next doubleword down, which needs separate address logic.
define i64 @f6(i64 %dummy, ptr %src, i64 %b) {
; CHECK-LABEL: f6:
; CHECK: agfi %r3, -524296
; CHECK: lang %r2, %r4, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -65537
  %res = atomicrmw and ptr %ptr, i64 %b seq_cst
  ret i64 %res
}
