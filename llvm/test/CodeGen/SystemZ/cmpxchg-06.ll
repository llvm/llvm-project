; Test 64-bit compare and swap.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check CDSG without a displacement.
define i128 @f1(i128 %cmp, i128 %swap, i128 *%src) {
; CHECK-LABEL: f1:
; CHECK-DAG: lg %r1, 8(%r4)
; CHECK-DAG: lg %r0, 0(%r4)
; CHECK-DAG: lg %r13, 8(%r3)
; CHECK-DAG: lg %r12, 0(%r3)
; CHECK:     cdsg %r12, %r0, 0(%r5)
; CHECK-DAG: stg %r13, 8(%r2)
; CHECK-DAG: stg %r12, 0(%r2)
; CHECK:     br %r14
  %pairval = cmpxchg i128 *%src, i128 %cmp, i128 %swap seq_cst seq_cst
  %val = extractvalue { i128, i1 } %pairval, 0
  ret i128 %val
}

; Check the high end of the aligned CDSG range.
define i128 @f2(i128 %cmp, i128 %swap, i128 *%src) {
; CHECK-LABEL: f2:
; CHECK: cdsg {{%r[0-9]+}}, {{%r[0-9]+}}, 524272(%r5)
; CHECK: br %r14
  %ptr = getelementptr i128, i128 *%src, i128 32767
  %pairval = cmpxchg i128 *%ptr, i128 %cmp, i128 %swap seq_cst seq_cst
  %val = extractvalue { i128, i1 } %pairval, 0
  ret i128 %val
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i128 @f3(i128 %cmp, i128 %swap, i128 *%src) {
; CHECK-LABEL: f3:
; CHECK: agfi %r5, 524288
; CHECK: cdsg {{%r[0-9]+}}, {{%r[0-9]+}}, 0(%r5)
; CHECK: br %r14
  %ptr = getelementptr i128, i128 *%src, i128 32768
  %pairval = cmpxchg i128 *%ptr, i128 %cmp, i128 %swap seq_cst seq_cst
  %val = extractvalue { i128, i1 } %pairval, 0
  ret i128 %val
}

; Check the high end of the negative aligned CDSG range.
define i128 @f4(i128 %cmp, i128 %swap, i128 *%src) {
; CHECK-LABEL: f4:
; CHECK: cdsg {{%r[0-9]+}}, {{%r[0-9]+}}, -16(%r5)
; CHECK: br %r14
  %ptr = getelementptr i128, i128 *%src, i128 -1
  %pairval = cmpxchg i128 *%ptr, i128 %cmp, i128 %swap seq_cst seq_cst
  %val = extractvalue { i128, i1 } %pairval, 0
  ret i128 %val
}

; Check the low end of the CDSG range.
define i128 @f5(i128 %cmp, i128 %swap, i128 *%src) {
; CHECK-LABEL: f5:
; CHECK: cdsg {{%r[0-9]+}}, {{%r[0-9]+}}, -524288(%r5)
; CHECK: br %r14
  %ptr = getelementptr i128, i128 *%src, i128 -32768
  %pairval = cmpxchg i128 *%ptr, i128 %cmp, i128 %swap seq_cst seq_cst
  %val = extractvalue { i128, i1 } %pairval, 0
  ret i128 %val
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i128 @f6(i128 %cmp, i128 %swap, i128 *%src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r5, -524304
; CHECK: cdsg {{%r[0-9]+}}, {{%r[0-9]+}}, 0(%r5)
; CHECK: br %r14
  %ptr = getelementptr i128, i128 *%src, i128 -32769
  %pairval = cmpxchg i128 *%ptr, i128 %cmp, i128 %swap seq_cst seq_cst
  %val = extractvalue { i128, i1 } %pairval, 0
  ret i128 %val
}

; Check that CDSG does not allow an index.
define i128 @f7(i128 %cmp, i128 %swap, i64 %src, i64 %index) {
; CHECK-LABEL: f7:
; CHECK: agr %r5, %r6
; CHECK: cdsg {{%r[0-9]+}}, {{%r[0-9]+}}, 0(%r5)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %ptr = inttoptr i64 %add1 to i128 *
  %pairval = cmpxchg i128 *%ptr, i128 %cmp, i128 %swap seq_cst seq_cst
  %val = extractvalue { i128, i1 } %pairval, 0
  ret i128 %val
}

; Check that a constant %cmp value is loaded into a register first.
define i128 @f8(i128 %swap, i128 *%ptr) {
; CHECK-LABEL: f8:
; CHECK: lghi {{%r[0-9]+}}, 1001
; CHECK: cdsg {{%r[0-9]+}}, {{%r[0-9]+}}, 0(%r4)
; CHECK: br %r14
  %pairval = cmpxchg i128 *%ptr, i128 1001, i128 %swap seq_cst seq_cst
  %val = extractvalue { i128, i1 } %pairval, 0
  ret i128 %val
}

; Check that a constant %swap value is loaded into a register first.
define i128 @f9(i128 %cmp, i128 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: lghi {{%r[0-9]+}}, 1002
; CHECK: cdsg {{%r[0-9]+}}, {{%r[0-9]+}}, 0(%r4)
; CHECK: br %r14
  %pairval = cmpxchg i128 *%ptr, i128 %cmp, i128 1002 seq_cst seq_cst
  %val = extractvalue { i128, i1 } %pairval, 0
  ret i128 %val
}

; Check generating the comparison result.
; CHECK-LABEL: f10
; CHECK-DAG: lg %r1, 8(%r3)
; CHECK-DAG: lg %r0, 0(%r3)
; CHECK-DAG: lg %r13, 8(%r2)
; CHECK-DAG: lg %r12, 0(%r2)
; CHECK:     cdsg %r12, %r0, 0(%r4)
; CHECK-NEXT: ipm %r2
; CHECK-NEXT: afi %r2, -268435456
; CHECK-NEXT: srl %r2, 31
; CHECK: br %r14
define i32 @f10(i128 %cmp, i128 %swap, i128 *%src) {
  %pairval = cmpxchg i128 *%src, i128 %cmp, i128 %swap seq_cst seq_cst
  %val = extractvalue { i128, i1 } %pairval, 1
  %res = zext i1 %val to i32
  ret i32 %res
}
