; Test 32-bit atomic subtractions, z196 version.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check addition of a variable.
define i32 @f1(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: lcr [[NEG:%r[0-5]]], %r4
; CHECK: laa %r2, [[NEG]], 0(%r3)
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i32 %b seq_cst
  ret i32 %res
}

; Check addition of 1, which needs a temporary.
define i32 @f2(i32 %dummy, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: lhi [[TMP:%r[0-5]]], -1
; CHECK: laa %r2, [[TMP]], 0(%r3)
; CHECK: br %r14
  %res = atomicrmw sub ptr %src, i32 1 seq_cst
  ret i32 %res
}

; Check the high end of the LAA range.
define i32 @f3(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: lcr [[NEG:%r[0-5]]], %r4
; CHECK: laa %r2, [[NEG]], 524284(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i32 131071
  %res = atomicrmw sub ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the next word up, which needs separate address logic.
define i32 @f4(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f4:
; CHECK-DAG: lcr [[NEG:%r[0-5]]], %r4
; CHECK-DAG: agfi %r3, 524288
; CHECK: laa %r2, [[NEG]], 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i32 131072
  %res = atomicrmw sub ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the low end of the LAA range.
define i32 @f5(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f5:
; CHECK: lcr [[NEG:%r[0-5]]], %r4
; CHECK: laa %r2, [[NEG]], -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i32 -131072
  %res = atomicrmw sub ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the next word down, which needs separate address logic.
define i32 @f6(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f6:
; CHECK-DAG: lcr [[NEG:%r[0-5]]], %r4
; CHECK-DAG: agfi %r3, -524292
; CHECK: laa %r2, [[NEG]], 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i32 -131073
  %res = atomicrmw sub ptr %ptr, i32 %b seq_cst
  ret i32 %res
}
