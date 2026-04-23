; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s

; Test coverage for HexagonGenPredicate: exercise various predicate
; conversion paths including A4_andn/C2_andn, A4_orn/C2_orn,
; C2_cmpeqi==0 to C2_not, C4_cmpneqi==0 to COPY, and deeper
; predicate chains using various comparison types.

; CHECK-LABEL: test_pred_andn:
; CHECK: cmp
define i32 @test_pred_andn(i32 %a, i32 %b, i32 %c) {
entry:
  %cmp1 = icmp sgt i32 %a, 0
  %cmp2 = icmp sgt i32 %b, 0
  %not2 = xor i1 %cmp2, true
  %and = and i1 %cmp1, %not2
  %sel = select i1 %and, i32 %a, i32 %c
  ret i32 %sel
}

; CHECK-LABEL: test_pred_orn:
; CHECK: cmp
define i32 @test_pred_orn(i32 %a, i32 %b, i32 %c) {
entry:
  %cmp1 = icmp sgt i32 %a, 0
  %cmp2 = icmp sgt i32 %b, 0
  %not2 = xor i1 %cmp2, true
  %or = or i1 %cmp1, %not2
  %sel = select i1 %or, i32 %a, i32 %c
  ret i32 %sel
}

; Exercise a more complex predicate chain involving and_and (C4_and_and)
; and or_or (C4_or_or).
; CHECK-LABEL: test_pred_chain:
; CHECK: cmp
define i32 @test_pred_chain(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %cmp1 = icmp sgt i32 %a, 0
  %cmp2 = icmp sgt i32 %b, 0
  %cmp3 = icmp sgt i32 %c, 0
  %and12 = and i1 %cmp1, %cmp2
  %and123 = and i1 %and12, %cmp3
  %sel = select i1 %and123, i32 %d, i32 0
  ret i32 %sel
}

; Predicate comparison against 0: exercise C2_cmpeqi path.
; CHECK-LABEL: test_pred_cmp0:
; CHECK: cmp
define i32 @test_pred_cmp0(i32 %a, i32 %b) {
entry:
  %cmp = icmp eq i32 %a, 0
  %ext = zext i1 %cmp to i32
  %and = and i32 %ext, %b
  ret i32 %and
}

; Test cmpeq with 0 feeding into predicate logic (triggers C2_cmpeqi
; == 0 conversion path).
; CHECK-LABEL: test_cmpeq_zero:
; CHECK: cmp
define i32 @test_cmpeq_zero(i32 %a, i32 %b, i32 %x, i32 %y) {
entry:
  %c1 = icmp eq i32 %a, 0
  %c2 = icmp sgt i32 %b, 0
  %and = and i1 %c1, %c2
  %sel = select i1 %and, i32 %x, i32 %y
  ret i32 %sel
}

; Test ne with 0 feeding into predicate logic (triggers C4_cmpneqi
; == 0 conversion path).
; CHECK-LABEL: test_cmpne_zero:
; CHECK: cmp
define i32 @test_cmpne_zero(i32 %a, i32 %b, i32 %x, i32 %y) {
entry:
  %c1 = icmp ne i32 %a, 0
  %c2 = icmp sgt i32 %b, 0
  %and = and i1 %c1, %c2
  %sel = select i1 %and, i32 %x, i32 %y
  ret i32 %sel
}

; Test deeper predicate chain with multiple comparison types:
; signed ge and signed le (exercises C2_cmpgei / C4_cmpltei paths).
; CHECK-LABEL: test_deep_pred_chain:
; CHECK: cmp
define i32 @test_deep_pred_chain(i32 %a, i32 %b, i32 %c, i32 %d, i32 %x, i32 %y) {
entry:
  %c1 = icmp sge i32 %a, 10
  %c2 = icmp sle i32 %b, 20
  %c3 = icmp uge i32 %c, 5
  %c4 = icmp sgt i32 %d, 0
  %and1 = and i1 %c1, %c2
  %and2 = and i1 %c3, %c4
  %or = or i1 %and1, %and2
  %sel = select i1 %or, i32 %x, i32 %y
  ret i32 %sel
}

; Test predicate OR-NOT chain (exercises C2_orn path in GenPredicate).
; CHECK-LABEL: test_pred_orn_chain:
; CHECK: cmp
define i32 @test_pred_orn_chain(i32 %a, i32 %b, i32 %c, i32 %x, i32 %y) {
entry:
  %c1 = icmp sgt i32 %a, 0
  %c2 = icmp sgt i32 %b, 0
  %c3 = icmp eq i32 %c, 0
  %not = xor i1 %c2, true
  %or = or i1 %c1, %not
  %and = and i1 %or, %c3
  %sel = select i1 %and, i32 %x, i32 %y
  ret i32 %sel
}

; Test byte comparisons to exercise A4_cmpb* in isScalarCmp.
; CHECK-LABEL: test_byte_cmp:
; CHECK: cmp
define i32 @test_byte_cmp(i32 %a, i32 %b, i32 %x, i32 %y) {
entry:
  %a8 = and i32 %a, 255
  %b8 = and i32 %b, 255
  %c1 = icmp eq i32 %a8, %b8
  %c2 = icmp ugt i32 %a8, 10
  %and = and i1 %c1, %c2
  %sel = select i1 %and, i32 %x, i32 %y
  ret i32 %sel
}

