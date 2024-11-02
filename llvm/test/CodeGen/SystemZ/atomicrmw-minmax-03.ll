; Test 32-bit atomic minimum and maximum.  Here we match the z10 versions,
; which can't use LOCR.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check signed minium.
define i32 @f1(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: l %r2, 0(%r3)
; CHECK: j [[LOOP:\.[^:]*]]
; CHECK: [[BB1:\.[^:]*]]:
; CHECK: cs %r2, [[NEW:%r[0-9]+]], 0(%r3)
; CHECK: ber %r14
; CHECK: [[LOOP]]:
; CHECK: lr [[NEW]], %r2
; CHECK: crjle %r2, %r4, [[KEEP:\..*]]
; CHECK: lr [[NEW]], %r4
; CHECK: j [[BB1]]
  %res = atomicrmw min ptr %src, i32 %b seq_cst
  ret i32 %res
}

; Check signed maximum.
define i32 @f2(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: l %r2, 0(%r3)
; CHECK: j [[LOOP:\.[^:]*]]
; CHECK: [[BB1:\.[^:]*]]:
; CHECK: cs %r2, [[NEW:%r[0-9]+]], 0(%r3)
; CHECK: ber %r14
; CHECK: [[LOOP]]:
; CHECK: lr [[NEW]], %r2
; CHECK: crjhe %r2, %r4, [[KEEP:\..*]]
; CHECK: lr [[NEW]], %r4
; CHECK: j [[BB1]]
  %res = atomicrmw max ptr %src, i32 %b seq_cst
  ret i32 %res
}

; Check unsigned minimum.
define i32 @f3(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: l %r2, 0(%r3)
; CHECK: j [[LOOP:\.[^:]*]]
; CHECK: [[BB1:\.[^:]*]]:
; CHECK: cs %r2, [[NEW:%r[0-9]+]], 0(%r3)
; CHECK: ber %r14
; CHECK: [[LOOP]]:
; CHECK: lr [[NEW]], %r2
; CHECK: clrjle %r2, %r4, [[KEEP:\..*]]
; CHECK: lr [[NEW]], %r4
; CHECK: j [[BB1]]
  %res = atomicrmw umin ptr %src, i32 %b seq_cst
  ret i32 %res
}

; Check unsigned maximum.
define i32 @f4(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f4:
; CHECK: l %r2, 0(%r3)
; CHECK: j [[LOOP:\.[^:]*]]
; CHECK: [[BB1:\.[^:]*]]:
; CHECK: cs %r2, [[NEW:%r[0-9]+]], 0(%r3)
; CHECK: ber %r14
; CHECK: [[LOOP]]:
; CHECK: lr [[NEW]], %r2
; CHECK: clrjhe %r2, %r4, [[KEEP:\..*]]
; CHECK: lr [[NEW]], %r4
; CHECK: j [[BB1]]
  %res = atomicrmw umax ptr %src, i32 %b seq_cst
  ret i32 %res
}

; Check the high end of the aligned CS range.
define i32 @f5(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f5:
; CHECK: l %r2, 4092(%r3)
; CHECK: cs %r2, {{%r[0-9]+}}, 4092(%r3)
; CHECK: ber %r14
  %ptr = getelementptr i32, ptr %src, i64 1023
  %res = atomicrmw min ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the next word up, which requires CSY.
define i32 @f6(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f6:
; CHECK: ly %r2, 4096(%r3)
; CHECK: csy %r2, {{%r[0-9]+}}, 4096(%r3)
; CHECK: ber %r14
  %ptr = getelementptr i32, ptr %src, i64 1024
  %res = atomicrmw min ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the high end of the aligned CSY range.
define i32 @f7(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f7:
; CHECK: ly %r2, 524284(%r3)
; CHECK: csy %r2, {{%r[0-9]+}}, 524284(%r3)
; CHECK: ber %r14
  %ptr = getelementptr i32, ptr %src, i64 131071
  %res = atomicrmw min ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the next word up, which needs separate address logic.
define i32 @f8(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f8:
; CHECK: agfi %r3, 524288
; CHECK: l %r2, 0(%r3)
; CHECK: cs %r2, {{%r[0-9]+}}, 0(%r3)
; CHECK: ber %r14
  %ptr = getelementptr i32, ptr %src, i64 131072
  %res = atomicrmw min ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the high end of the negative aligned CSY range.
define i32 @f9(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f9:
; CHECK: ly %r2, -4(%r3)
; CHECK: csy %r2, {{%r[0-9]+}}, -4(%r3)
; CHECK: ber %r14
  %ptr = getelementptr i32, ptr %src, i64 -1
  %res = atomicrmw min ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the low end of the CSY range.
define i32 @f10(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f10:
; CHECK: ly %r2, -524288(%r3)
; CHECK: csy %r2, {{%r[0-9]+}}, -524288(%r3)
; CHECK: ber %r14
  %ptr = getelementptr i32, ptr %src, i64 -131072
  %res = atomicrmw min ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check the next word down, which needs separate address logic.
define i32 @f11(i32 %dummy, ptr %src, i32 %b) {
; CHECK-LABEL: f11:
; CHECK: agfi %r3, -524292
; CHECK: l %r2, 0(%r3)
; CHECK: cs %r2, {{%r[0-9]+}}, 0(%r3)
; CHECK: ber %r14
  %ptr = getelementptr i32, ptr %src, i64 -131073
  %res = atomicrmw min ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check that indexed addresses are not allowed.
define i32 @f12(i32 %dummy, i64 %base, i64 %index, i32 %b) {
; CHECK-LABEL: f12:
; CHECK: agr %r3, %r4
; CHECK: l %r2, 0(%r3)
; CHECK: cs %r2, {{%r[0-9]+}}, 0(%r3)
; CHECK: ber %r14
  %add = add i64 %base, %index
  %ptr = inttoptr i64 %add to ptr
  %res = atomicrmw min ptr %ptr, i32 %b seq_cst
  ret i32 %res
}

; Check that constants are handled.
define i32 @f13(i32 %dummy, ptr %ptr) {
; CHECK-LABEL: f13:
; CHECK: lhi [[LIMIT:%r[0-9]+]], 42
; CHECK: j [[LOOP:\.[^:]*]]
; CHECK: [[BB1:\.[^:]*]]:
; CHECK: cs %r2, [[NEW:%r[0-9]+]], 0(%r3)
; CHECK: ber %r14
; CHECK: [[LOOP]]:
; CHECK: lr [[NEW]], %r2
; CHECK: crjle %r2, [[LIMIT]], [[KEEP:\..*]]
; CHECK: lhi [[NEW]], 42
; CHECK: j [[BB1]]
  %res = atomicrmw min ptr %ptr, i32 42 seq_cst
  ret i32 %res
}
