; Test 64-bit compare and swap.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check CSG without a displacement.
define i64 @f1(i64 %cmp, i64 %swap, i64 *%src) {
; CHECK-LABEL: f1:
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK: br %r14
  %pairval = cmpxchg i64 *%src, i64 %cmp, i64 %swap seq_cst seq_cst
  %val = extractvalue { i64, i1 } %pairval, 0
  ret i64 %val
}

; Check the high end of the aligned CSG range.
define i64 @f2(i64 %cmp, i64 %swap, i64 *%src) {
; CHECK-LABEL: f2:
; CHECK: csg %r2, %r3, 524280(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65535
  %pairval = cmpxchg i64 *%ptr, i64 %cmp, i64 %swap seq_cst seq_cst
  %val = extractvalue { i64, i1 } %pairval, 0
  ret i64 %val
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f3(i64 %cmp, i64 %swap, i64 *%src) {
; CHECK-LABEL: f3:
; CHECK: agfi %r4, 524288
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65536
  %pairval = cmpxchg i64 *%ptr, i64 %cmp, i64 %swap seq_cst seq_cst
  %val = extractvalue { i64, i1 } %pairval, 0
  ret i64 %val
}

; Check the high end of the negative aligned CSG range.
define i64 @f4(i64 %cmp, i64 %swap, i64 *%src) {
; CHECK-LABEL: f4:
; CHECK: csg %r2, %r3, -8(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -1
  %pairval = cmpxchg i64 *%ptr, i64 %cmp, i64 %swap seq_cst seq_cst
  %val = extractvalue { i64, i1 } %pairval, 0
  ret i64 %val
}

; Check the low end of the CSG range.
define i64 @f5(i64 %cmp, i64 %swap, i64 *%src) {
; CHECK-LABEL: f5:
; CHECK: csg %r2, %r3, -524288(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65536
  %pairval = cmpxchg i64 *%ptr, i64 %cmp, i64 %swap seq_cst seq_cst
  %val = extractvalue { i64, i1 } %pairval, 0
  ret i64 %val
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f6(i64 %cmp, i64 %swap, i64 *%src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r4, -524296
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65537
  %pairval = cmpxchg i64 *%ptr, i64 %cmp, i64 %swap seq_cst seq_cst
  %val = extractvalue { i64, i1 } %pairval, 0
  ret i64 %val
}

; Check that CSG does not allow an index.
define i64 @f7(i64 %cmp, i64 %swap, i64 %src, i64 %index) {
; CHECK-LABEL: f7:
; CHECK: agr %r4, %r5
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %ptr = inttoptr i64 %add1 to i64 *
  %pairval = cmpxchg i64 *%ptr, i64 %cmp, i64 %swap seq_cst seq_cst
  %val = extractvalue { i64, i1 } %pairval, 0
  ret i64 %val
}

; Check that a constant %cmp value is loaded into a register first.
define i64 @f8(i64 %dummy, i64 %swap, i64 *%ptr) {
; CHECK-LABEL: f8:
; CHECK: lghi %r2, 1001
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK: br %r14
  %pairval = cmpxchg i64 *%ptr, i64 1001, i64 %swap seq_cst seq_cst
  %val = extractvalue { i64, i1 } %pairval, 0
  ret i64 %val
}

; Check that a constant %swap value is loaded into a register first.
define i64 @f9(i64 %cmp, i64 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: lghi [[SWAP:%r[0-9]+]], 1002
; CHECK: csg %r2, [[SWAP]], 0(%r3)
; CHECK: br %r14
  %pairval = cmpxchg i64 *%ptr, i64 %cmp, i64 1002 seq_cst seq_cst
  %val = extractvalue { i64, i1 } %pairval, 0
  ret i64 %val
}

; Check generating the comparison result.
; CHECK-LABEL: f10
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK-NEXT: ipm %r2
; CHECK-NEXT: afi %r2, -268435456
; CHECK-NEXT: srl %r2, 31
; CHECK: br %r14
define i32 @f10(i64 %cmp, i64 %swap, i64 *%src) {
  %pairval = cmpxchg i64 *%src, i64 %cmp, i64 %swap seq_cst seq_cst
  %val = extractvalue { i64, i1 } %pairval, 1
  %res = zext i1 %val to i32
  ret i32 %res
}

declare void @g()

; Check using the comparison result for a branch.
; CHECK-LABEL: f11
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK-NEXT: jge g
; CHECK: br %r14
define void @f11(i64 %cmp, i64 %swap, i64 *%src) {
  %pairval = cmpxchg i64 *%src, i64 %cmp, i64 %swap seq_cst seq_cst
  %cond = extractvalue { i64, i1 } %pairval, 1
  br i1 %cond, label %call, label %exit

call:
  tail call void @g()
  br label %exit

exit:
  ret void
}

; ... and the same with the inverted direction.
; CHECK-LABEL: f12
; CHECK: csg %r2, %r3, 0(%r4)
; CHECK-NEXT: jgl g
; CHECK: br %r14
define void @f12(i64 %cmp, i64 %swap, i64 *%src) {
  %pairval = cmpxchg i64 *%src, i64 %cmp, i64 %swap seq_cst seq_cst
  %cond = extractvalue { i64, i1 } %pairval, 1
  br i1 %cond, label %exit, label %call

call:
  tail call void @g()
  br label %exit

exit:
  ret void
}

