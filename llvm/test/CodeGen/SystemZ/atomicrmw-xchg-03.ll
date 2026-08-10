; Test 32-bit atomic exchange.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check register exchange.
define i32 @f1(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: l %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: cs %r2, %r4, 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw xchg ptr %src, i32 %b seq_cst
  ret i32 %res
}

; Check the high end of the aligned CS range.
define i32 @f2(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: l %r2, 4092(%r3)
; CHECK: cs %r2, {{%r[0-9]+}}, 4092(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 1023
  %res = atomicrmw xchg ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the next word up, which requires CSY.
define i32 @f3(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: ly %r2, 4096(%r3)
; CHECK: csy %r2, {{%r[0-9]+}}, 4096(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 1024
  %res = atomicrmw xchg ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the high end of the aligned CSY range.
define i32 @f4(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f4:
; CHECK: ly %r2, 524284(%r3)
; CHECK: csy %r2, {{%r[0-9]+}}, 524284(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131071
  %res = atomicrmw xchg ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the next word up, which needs separate address logic.
define i32 @f5(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f5:
; CHECK: agfi %r3, 524288
; CHECK: l %r2, 0(%r3)
; CHECK: cs %r2, {{%r[0-9]+}}, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131072
  %res = atomicrmw xchg ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the high end of the negative aligned CSY range.
define i32 @f6(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f6:
; CHECK: ly %r2, -4(%r3)
; CHECK: csy %r2, {{%r[0-9]+}}, -4(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -1
  %res = atomicrmw xchg ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the low end of the CSY range.
define i32 @f7(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f7:
; CHECK: ly %r2, -524288(%r3)
; CHECK: csy %r2, {{%r[0-9]+}}, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131072
  %res = atomicrmw xchg ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the next word down, which needs separate address logic.
define i32 @f8(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f8:
; CHECK: agfi %r3, -524292
; CHECK: l %r2, 0(%r3)
; CHECK: cs %r2, {{%r[0-9]+}}, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 -131073
  %res = atomicrmw xchg ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check that indexed addresses are not allowed.
define i32 @f9(i32 %dummy, i64 %base, i64 %index, i32 %b) {
; CHECK-LABEL: f9:
; CHECK: l %r2, 0(%r4,%r3)
; CHECK: agr %r3, %r4
; CHECK: cs %r2, {{%r[0-9]+}}, 0(%r3)
; CHECK: br %r14
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to ptr
  %res = atomicrmw xchg ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check exchange of a constant.  We should force it into a register and
; use the sequence above.
define i32 @f10(i32 %dummy, ptr %src) {
; CHECK-LABEL: f10:
; CHECK-DAG: llill [[VALUE:%r[0-9+]]], 40000
; CHECK-DAG: l %r2, 0(%r3)
; CHECK: [[LABEL:\.[^:]*]]:
; CHECK: cs %r2, [[VALUE]], 0(%r3)
; CHECK: jl [[LABEL]]
; CHECK: br %r14
  %res = atomicrmw xchg ptr %src, i32 40000 seq_cst
  ret i32 %res
}

