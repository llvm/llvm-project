; Test 32-bit compare and swap.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the low end of the CS range.
define i32 @f1(i32 %cmp, i32 %swap, i32 *%src) {
; CHECK-LABEL: f1:
; CHECK: cs %r2, %r3, 0(%r4)
; CHECK: br %r14
  %pair = cmpxchg i32 *%src, i32 %cmp, i32 %swap seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; Check the high end of the aligned CS range.
define i32 @f2(i32 %cmp, i32 %swap, i32 *%src) {
; CHECK-LABEL: f2:
; CHECK: cs %r2, %r3, 4092(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 1023
  %pair = cmpxchg i32 *%ptr, i32 %cmp, i32 %swap seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; Check the next word up, which should use CSY instead of CS.
define i32 @f3(i32 %cmp, i32 %swap, i32 *%src) {
; CHECK-LABEL: f3:
; CHECK: csy %r2, %r3, 4096(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 1024
  %pair = cmpxchg i32 *%ptr, i32 %cmp, i32 %swap seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; Check the high end of the aligned CSY range.
define i32 @f4(i32 %cmp, i32 %swap, i32 *%src) {
; CHECK-LABEL: f4:
; CHECK: csy %r2, %r3, 524284(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131071
  %pair = cmpxchg i32 *%ptr, i32 %cmp, i32 %swap seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f5(i32 %cmp, i32 %swap, i32 *%src) {
; CHECK-LABEL: f5:
; CHECK: agfi %r4, 524288
; CHECK: cs %r2, %r3, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131072
  %pair = cmpxchg i32 *%ptr, i32 %cmp, i32 %swap seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; Check the high end of the negative aligned CSY range.
define i32 @f6(i32 %cmp, i32 %swap, i32 *%src) {
; CHECK-LABEL: f6:
; CHECK: csy %r2, %r3, -4(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -1
  %pair = cmpxchg i32 *%ptr, i32 %cmp, i32 %swap seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; Check the low end of the CSY range.
define i32 @f7(i32 %cmp, i32 %swap, i32 *%src) {
; CHECK-LABEL: f7:
; CHECK: csy %r2, %r3, -524288(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131072
  %pair = cmpxchg i32 *%ptr, i32 %cmp, i32 %swap seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f8(i32 %cmp, i32 %swap, i32 *%src) {
; CHECK-LABEL: f8:
; CHECK: agfi %r4, -524292
; CHECK: cs %r2, %r3, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131073
  %pair = cmpxchg i32 *%ptr, i32 %cmp, i32 %swap seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; Check that CS does not allow an index.
define i32 @f9(i32 %cmp, i32 %swap, i64 %src, i64 %index) {
; CHECK-LABEL: f9:
; CHECK: agr %r4, %r5
; CHECK: cs %r2, %r3, 0(%r4)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %ptr = inttoptr i64 %add1 to i32 *
  %pair = cmpxchg i32 *%ptr, i32 %cmp, i32 %swap seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; Check that CSY does not allow an index.
define i32 @f10(i32 %cmp, i32 %swap, i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: agr %r4, %r5
; CHECK: csy %r2, %r3, 4096(%r4)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i32 *
  %pair = cmpxchg i32 *%ptr, i32 %cmp, i32 %swap seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; Check that a constant %cmp value is loaded into a register first.
define i32 @f11(i32 %dummy, i32 %swap, i32 *%ptr) {
; CHECK-LABEL: f11:
; CHECK: lhi %r2, 1001
; CHECK: cs %r2, %r3, 0(%r4)
; CHECK: br %r14
  %pair = cmpxchg i32 *%ptr, i32 1001, i32 %swap seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; Check that a constant %swap value is loaded into a register first.
define i32 @f12(i32 %cmp, i32 *%ptr) {
; CHECK-LABEL: f12:
; CHECK: lhi [[SWAP:%r[0-9]+]], 1002
; CHECK: cs %r2, [[SWAP]], 0(%r3)
; CHECK: br %r14
  %pair = cmpxchg i32 *%ptr, i32 %cmp, i32 1002 seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pair, 0
  ret i32 %val
}

; Check generating the comparison result.
; CHECK-LABEL: f13
; CHECK: cs %r2, %r3, 0(%r4)
; CHECK-NEXT: ipm %r2
; CHECK-NEXT: afi %r2, -268435456
; CHECK-NEXT: srl %r2, 31
; CHECK: br %r14
define i32 @f13(i32 %cmp, i32 %swap, i32 *%src) {
  %pairval = cmpxchg i32 *%src, i32 %cmp, i32 %swap seq_cst seq_cst
  %val = extractvalue { i32, i1 } %pairval, 1
  %res = zext i1 %val to i32
  ret i32 %res
}
