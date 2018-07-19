; RUN: opt -instcombine -S -o - %s | FileCheck %s

define i1 @masked_and_notallzeroes(i32 %A) {
; CHECK-LABEL: @masked_and_notallzeroes(
; CHECK-NEXT:    [[MASK1:%.*]] = and i32 %A, 7
; CHECK-NEXT:    [[TST1:%.*]] = icmp ne i32 [[MASK1]], 0
; CHECK-NEXT:    ret i1 [[TST1]]
;
  %mask1 = and i32 %A, 7
  %tst1 = icmp ne i32 %mask1, 0
  %mask2 = and i32 %A, 39
  %tst2 = icmp ne i32 %mask2, 0
  %res = and i1 %tst1, %tst2
  ret i1 %res
}

define i1 @masked_or_allzeroes(i32 %A) {
; CHECK-LABEL: @masked_or_allzeroes(
; CHECK-NEXT:    [[MASK1:%.*]] = and i32 %A, 7
; CHECK-NEXT:    [[TST1:%.*]] = icmp eq i32 [[MASK1]], 0
; CHECK-NEXT:    ret i1 [[TST1]]
;
  %mask1 = and i32 %A, 7
  %tst1 = icmp eq i32 %mask1, 0
  %mask2 = and i32 %A, 39
  %tst2 = icmp eq i32 %mask2, 0
  %res = or i1 %tst1, %tst2
  ret i1 %res
}

define i1 @masked_and_notallones(i32 %A) {
; CHECK-LABEL: @masked_and_notallones(
; CHECK-NEXT:    [[MASK1:%.*]] = and i32 %A, 7
; CHECK-NEXT:    [[TST1:%.*]] = icmp ne i32 [[MASK1]], 7
; CHECK-NEXT:    ret i1 [[TST1]]
;
  %mask1 = and i32 %A, 7
  %tst1 = icmp ne i32 %mask1, 7
  %mask2 = and i32 %A, 39
  %tst2 = icmp ne i32 %mask2, 39
  %res = and i1 %tst1, %tst2
  ret i1 %res
}

define i1 @masked_or_allones(i32 %A) {
; CHECK-LABEL: @masked_or_allones(
; CHECK-NEXT:    [[MASK1:%.*]] = and i32 %A, 7
; CHECK-NEXT:    [[TST1:%.*]] = icmp eq i32 [[MASK1]], 7
; CHECK-NEXT:    ret i1 [[TST1]]
;
  %mask1 = and i32 %A, 7
  %tst1 = icmp eq i32 %mask1, 7
  %mask2 = and i32 %A, 39
  %tst2 = icmp eq i32 %mask2, 39
  %res = or i1 %tst1, %tst2
  ret i1 %res
}

define i1 @masked_and_notA(i32 %A) {
; CHECK-LABEL: @masked_and_notA(
; CHECK-NEXT:    [[MASK2:%.*]] = and i32 %A, 39
; CHECK-NEXT:    [[TST2:%.*]] = icmp ne i32 [[MASK2]], %A
; CHECK-NEXT:    ret i1 [[TST2]]
;
  %mask1 = and i32 %A, 7
  %tst1 = icmp ne i32 %mask1, %A
  %mask2 = and i32 %A, 39
  %tst2 = icmp ne i32 %mask2, %A
  %res = and i1 %tst1, %tst2
  ret i1 %res
}

define i1 @masked_or_A(i32 %A) {
; CHECK-LABEL: @masked_or_A(
; CHECK-NEXT:    [[MASK2:%.*]] = and i32 %A, 39
; CHECK-NEXT:    [[TST2:%.*]] = icmp eq i32 [[MASK2]], %A
; CHECK-NEXT:    ret i1 [[TST2]]
;
  %mask1 = and i32 %A, 7
  %tst1 = icmp eq i32 %mask1, %A
  %mask2 = and i32 %A, 39
  %tst2 = icmp eq i32 %mask2, %A
  %res = or i1 %tst1, %tst2
  ret i1 %res
}

define i1 @masked_or_allzeroes_notoptimised(i32 %A) {
; CHECK-LABEL: @masked_or_allzeroes_notoptimised(
; CHECK-NEXT:    [[MASK1:%.*]] = and i32 %A, 15
; CHECK-NEXT:    [[TST1:%.*]] = icmp eq i32 [[MASK1]], 0
; CHECK-NEXT:    [[MASK2:%.*]] = and i32 %A, 39
; CHECK-NEXT:    [[TST2:%.*]] = icmp eq i32 [[MASK2]], 0
; CHECK-NEXT:    [[RES:%.*]] = or i1 [[TST1]], [[TST2]]
; CHECK-NEXT:    ret i1 [[RES]]
;
  %mask1 = and i32 %A, 15
  %tst1 = icmp eq i32 %mask1, 0
  %mask2 = and i32 %A, 39
  %tst2 = icmp eq i32 %mask2, 0
  %res = or i1 %tst1, %tst2
  ret i1 %res
}

define i1 @nomask_lhs(i32 %in) {
; CHECK-LABEL: @nomask_lhs(
; CHECK-NEXT:    [[MASKED:%.*]] = and i32 %in, 1
; CHECK-NEXT:    [[TST2:%.*]] = icmp eq i32 [[MASKED]], 0
; CHECK-NEXT:    ret i1 [[TST2]]
;
  %tst1 = icmp eq i32 %in, 0
  %masked = and i32 %in, 1
  %tst2 = icmp eq i32 %masked, 0
  %val = or i1 %tst1, %tst2
  ret i1 %val
}

define i1 @nomask_rhs(i32 %in) {
; CHECK-LABEL: @nomask_rhs(
; CHECK-NEXT:    [[MASKED:%.*]] = and i32 %in, 1
; CHECK-NEXT:    [[TST1:%.*]] = icmp eq i32 [[MASKED]], 0
; CHECK-NEXT:    ret i1 [[TST1]]
;
  %masked = and i32 %in, 1
  %tst1 = icmp eq i32 %masked, 0
  %tst2 = icmp eq i32 %in, 0
  %val = or i1 %tst1, %tst2
  ret i1 %val
}

; TODO: This test simplifies to a constant, so the functionality and test could be in InstSimplify.

define i1 @fold_mask_cmps_to_false(i32 %x) {
; CHECK-LABEL: @fold_mask_cmps_to_false(
; CHECK-NEXT:    ret i1 false
;
  %1 = and i32 %x, 2147483647
  %2 = icmp eq i32 %1, 0
  %3 = icmp eq i32 %x, 2147483647
  %4 = and i1 %3, %2
  ret i1 %4
}

; TODO: This test simplifies to a constant, so the functionality and test could be in InstSimplify.

define i1 @fold_mask_cmps_to_true(i32 %x) {
; CHECK-LABEL: @fold_mask_cmps_to_true(
; CHECK-NEXT:    ret i1 true
;
  %1 = and i32 %x, 2147483647
  %2 = icmp ne i32 %1, 0
  %3 = icmp ne i32 %x, 2147483647
  %4 = or i1 %3, %2
  ret i1 %4
}

; PR32401 - https://bugs.llvm.org/show_bug.cgi?id=32401

define i1 @cmpeq_bitwise(i8 %a, i8 %b, i8 %c, i8 %d) {
; CHECK-LABEL: @cmpeq_bitwise(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp eq i8 %a, %b
; CHECK-NEXT:    [[TMP2:%.*]] = icmp eq i8 %c, %d
; CHECK-NEXT:    [[CMP:%.*]] = and i1 [[TMP1]], [[TMP2]]
; CHECK-NEXT:    ret i1 [[CMP]]
;
  %xor1 = xor i8 %a, %b
  %xor2 = xor i8 %c, %d
  %or = or i8 %xor1, %xor2
  %cmp = icmp eq i8 %or, 0
  ret i1 %cmp
}

define <2 x i1> @cmpne_bitwise(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c, <2 x i64> %d) {
; CHECK-LABEL: @cmpne_bitwise(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ne <2 x i64> %a, %b
; CHECK-NEXT:    [[TMP2:%.*]] = icmp ne <2 x i64> %c, %d
; CHECK-NEXT:    [[CMP:%.*]] = or <2 x i1> [[TMP1]], [[TMP2]]
; CHECK-NEXT:    ret <2 x i1> [[CMP]]
;
  %xor1 = xor <2 x i64> %a, %b
  %xor2 = xor <2 x i64> %c, %d
  %or = or <2 x i64> %xor1, %xor2
  %cmp = icmp ne <2 x i64> %or, zeroinitializer
  ret <2 x i1> %cmp
}

