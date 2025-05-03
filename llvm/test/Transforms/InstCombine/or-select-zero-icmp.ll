; RUN: opt < %s -passes=instcombine -S | FileCheck %s

; Basic functional test
define i32 @basic(i32 %a, i32 %b) {
; CHECK-LABEL: @basic(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[A:%.*]], 0
; CHECK-NEXT:    [[RES:%.*]] = select i1 [[CMP]], i32 [[B:%.*]], i32 [[A]]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %cmp = icmp eq i32 %a, 0
  %sel = select i1 %cmp, i32 %b, i32 0
  %or = or i32 %sel, %a
  ret i32 %or
}

; Operand order swap test
define i32 @swap_operand_order(i32 %x, i32 %y) {
; CHECK-LABEL: @swap_operand_order(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], 0
; CHECK-NEXT:    [[RES:%.*]] = select i1 [[CMP]], i32 [[Y:%.*]], i32 [[X]]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %cmp = icmp eq i32 %x, 0
  %sel = select i1 %cmp, i32 %y, i32 0
  %or = or i32 %x, %sel
  ret i32 %or
}

; Negative test: Non-zero false value in select
define i32 @negative_non_zero_false_val(i32 %a, i32 %b) {
; CHECK-LABEL: @negative_non_zero_false_val(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[A:%.*]], 0
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[CMP]], i32 [[B:%.*]], i32 1
; CHECK-NEXT:    [[OR:%.*]] = or i32 [[SEL]], [[A]]
; CHECK-NEXT:    ret i32 [[OR]]
;
  %cmp = icmp eq i32 %a, 0
  %sel = select i1 %cmp, i32 %b, i32 1
  %or = or i32 %sel, %a
  ret i32 %or
}

; Negative test: Incorrect comparison predicate (NE)
define i32 @negative_wrong_predicate(i32 %a, i32 %b) {
; CHECK-LABEL: @negative_wrong_predicate(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[A:%.*]], 0
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[CMP]], i32 0, i32 [[B:%.*]]
; CHECK-NEXT:    [[OR:%.*]] = or i32 [[SEL]], [[A]]
; CHECK-NEXT:    ret i32 [[OR]]
;
  %cmp = icmp ne i32 %a, 0
  %sel = select i1 %cmp, i32 %b, i32 0
  %or = or i32 %sel, %a
  ret i32 %or
}

; Comparison direction swap test (0 == X)
define i32 @cmp_swapped(i32 %x, i32 %y) {
; CHECK-LABEL: @cmp_swapped(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X:%.*]], 0
; CHECK-NEXT:    [[RES:%.*]] = select i1 [[CMP]], i32 [[Y:%.*]], i32 [[X]]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %cmp = icmp eq i32 0, %x
  %sel = select i1 %cmp, i32 %y, i32 0
  %or = or i32 %x, %sel
  ret i32 %or
}

; Complex expression test
define i32 @complex_expression(i32 %a, i32 %b) {
; CHECK-LABEL: @complex_expression(
; CHECK-NEXT:    [[X:%.*]] = add i32 [[A:%.*]], 1
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[X]], 0
; CHECK-NEXT:    [[RES:%.*]] = select i1 [[CMP]], i32 [[B:%.*]], i32 [[X]]
; CHECK-NEXT:    ret i32 [[RES]]
;
  %x = add i32 %a, 1
  %cmp = icmp eq i32 %x, 0
  %sel = select i1 %cmp, i32 %b, i32 0
  %or = or i32 %sel, %x
  ret i32 %or
}

; zext test
define i32 @zext_cond(i8 %a, i32 %b) {
; CHECK-LABEL: @zext_cond(
; CHECK-NEXT:    [[Z:%.*]] = zext i8 [[A:%.*]] to i32
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[A]], 0
; CHECK-NEXT:    [[OR:%.*]] = select i1 [[CMP]], i32 [[B:%.*]], i32 [[Z]]
; CHECK-NEXT:    ret i32 [[OR]]
  %z   = zext i8 %a to i32
  %cmp = icmp eq i32 %z, 0
  %sel = select i1 %cmp, i32 %b, i32 0
  %or  = or i32 %sel, %z
  ret i32 %or
}

; sext test
define i32 @sext_cond(i8 %a, i32 %b) {
; CHECK-LABEL: @sext_cond(
; CHECK-NEXT:    [[S:%.*]] = sext i8 [[A:%.*]] to i32
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[A]], 0
; CHECK-NEXT:    [[OR:%.*]] = select i1 [[CMP]], i32 [[B:%.*]], i32 [[S]]
; CHECK-NEXT:    ret i32 [[OR]]
  %s   = sext i8 %a to i32
  %cmp = icmp eq i32 %s, 0
  %sel = select i1 %cmp, i32 %b, i32 0
  %or  = or i32 %sel, %s
  ret i32 %or
}

; Vector type test
define <2 x i32> @vector_type(<2 x i32> %a, <2 x i32> %b) {
; CHECK-LABEL: @vector_type(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq <2 x i32> [[A:%.*]], zeroinitializer
; CHECK-NEXT:    [[RES:%.*]] = select <2 x i1> [[CMP]], <2 x i32> [[B:%.*]], <2 x i32> [[A]]
; CHECK-NEXT:    ret <2 x i32> [[RES]]
;
  %cmp = icmp eq <2 x i32> %a, zeroinitializer
  %sel = select <2 x i1> %cmp, <2 x i32> %b, <2 x i32> zeroinitializer
  %or = or <2 x i32> %sel, %a
  ret <2 x i32> %or
}

; Pointer type test (should not trigger optimization)
define i32* @pointer_type(i32* %p, i32* %q) {
; CHECK-LABEL: @pointer_type(
; CHECK-NEXT:    [[A:%.*]] = ptrtoint ptr [[P:%.*]] to i64
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr [[P]], null
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[CMP]], ptr [[Q:%.*]], ptr null
; CHECK-NEXT:    [[SEL_INT:%.*]] = ptrtoint ptr [[SEL]] to i64
; CHECK-NEXT:    [[OR:%.*]] = or i64 [[A]], [[SEL_INT]]
; CHECK-NEXT:    [[RET:%.*]] = inttoptr i64 [[OR]] to ptr
; CHECK-NEXT:    ret ptr [[RET]]
;
  %a = ptrtoint i32* %p to i64
  %cmp = icmp eq i64 %a, 0
  %sel = select i1 %cmp, i32* %q, i32* null
  %sel_int = ptrtoint i32* %sel to i64
  %or_val = or i64 %a, %sel_int
  %ret = inttoptr i64 %or_val to i32*
  ret i32* %ret
}
