; RUN: opt < %s -passes=debugify,instcombine -S | FileCheck %s
; RUN: opt < %s -passes=debugify,instcombine -S --try-experimental-debuginfo-iterators | FileCheck %s

declare void @escape32(i32)

define i64 @test_sext_zext(i16 %A) {
; CHECK-LABEL: @test_sext_zext(
; CHECK-NEXT:  [[C2:%.*]] = zext i16 %A to i64
; CHECK-NEXT:  #dbg_value(i64 [[C2]], {{.*}}, !DIExpression(),
; CHECK-NEXT:  #dbg_value(i64 [[C2]], {{.*}}, !DIExpression(),
  %c1 = zext i16 %A to i32
  %c2 = sext i32 %c1 to i64
  ret i64 %c2
}

define i64 @test_used_sext_zext(i16 %A) {
; CHECK-LABEL: @test_used_sext_zext(
; CHECK-NEXT:  [[C1:%.*]] = zext i16 %A to i32
; CHECK-NEXT:  #dbg_value(i32 [[C1]], {{.*}}, !DIExpression(),
; CHECK-NEXT:  [[C2:%.*]] = zext i16 %A to i64
; CHECK-NEXT:  #dbg_value(i64 [[C2]], {{.*}}, !DIExpression(),
; CHECK-NEXT:  call void @escape32(i32 %c1)
; CHECK-NEXT:  ret i64 %c2
  %c1 = zext i16 %A to i32
  %c2 = sext i32 %c1 to i64
  call void @escape32(i32 %c1)
  ret i64 %c2
}

define i32 @test_cast_select(i1 %cond) {
; CHECK-LABEL: @test_cast_select(
; CHECK-NEXT:  [[sel:%.*]] = select i1 %cond, i32 3, i32 5
; CHECK-NEXT:  #dbg_value(i32 [[sel]], {{.*}}, !DIExpression(),
; CHECK-NEXT:  #dbg_value(i32 [[sel]], {{.*}}, !DIExpression(),
; CHECK-NEXT:  ret i32 [[sel]]
  %sel = select i1 %cond, i16 3, i16 5
  %cast = zext i16 %sel to i32
  ret i32 %cast
}

define void @test_or(i64 %A) {
; CHECK-LABEL: @test_or(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 256, DW_OP_or, DW_OP_stack_value),
  %1 = or i64 %A, 256
  ret void
}

define void @test_xor(i32 %A) {
; CHECK-LABEL: @test_xor(
; CHECK-NEXT:  #dbg_value(i32 %A, {{.*}}, !DIExpression(DW_OP_constu, 1, DW_OP_xor, DW_OP_stack_value),
  %1 = xor i32 %A, 1
  ret void
}

define void @test_sub_neg(i64 %A) {
; CHECK-LABEL: @test_sub_neg(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value),
  %1 = sub i64 %A, -1
  ret void
}

define void @test_sub_pos(i64 %A) {
; CHECK-LABEL: @test_sub_pos(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value),
  %1 = sub i64 %A, 1
  ret void
}

define void @test_shl(i64 %A) {
; CHECK-LABEL: @test_shl(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 7, DW_OP_shl, DW_OP_stack_value),
  %1 = shl i64 %A, 7
  ret void
}

define void @test_lshr(i64 %A) {
; CHECK-LABEL: @test_lshr(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 7, DW_OP_shr, DW_OP_stack_value),
  %1 = lshr i64 %A, 7
  ret void
}

define void @test_ashr(i64 %A) {
; CHECK-LABEL: @test_ashr(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 7, DW_OP_shra, DW_OP_stack_value),
  %1 = ashr i64 %A, 7
  ret void
}

define void @test_mul(i64 %A) {
; CHECK-LABEL: @test_mul(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 7, DW_OP_mul, DW_OP_stack_value),
  %1 = mul i64 %A, 7
  ret void
}

define void @test_sdiv(i64 %A) {
; CHECK-LABEL: @test_sdiv(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 7, DW_OP_div, DW_OP_stack_value),
  %1 = sdiv i64 %A, 7
  ret void
}

define void @test_srem(i64 %A) {
; CHECK-LABEL: @test_srem(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 7, DW_OP_mod, DW_OP_stack_value),
  %1 = srem i64 %A, 7
  ret void
}

define void @test_ptrtoint(ptr %P) {
; CHECK-LABEL: @test_ptrtoint
; CHECK-NEXT:  #dbg_value(ptr %P, {{.*}}, !DIExpression(),
  %1 = ptrtoint ptr %P to i64
  ret void
}

define void @test_and(i64 %A) {
; CHECK-LABEL: @test_and(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 256, DW_OP_and, DW_OP_stack_value),
  %1 = and i64 %A, 256
  ret void
}
