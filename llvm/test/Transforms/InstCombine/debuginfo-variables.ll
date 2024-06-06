; RUN: opt < %s -passes=debugify,instcombine -S | FileCheck %s
; RUN: opt < %s -passes=debugify,instcombine -S --try-experimental-debuginfo-iterators | FileCheck %s

; RUN: opt < %s -passes=debugify,instcombine --debugify-diop-diexprs --experimental-debuginfo-iterators=true  -S | FileCheck %s --check-prefix DIOP-DBGINFO
; RUN: opt < %s -passes=debugify,instcombine --debugify-diop-diexprs --experimental-debuginfo-iterators=false -S | FileCheck %s --check-prefix DIOP-DBGINFO

declare void @escape32(i32)

define i64 @test_sext_zext(i16 %A) {
; CHECK-LABEL: @test_sext_zext(
; CHECK-NEXT:  [[C2:%.*]] = zext i16 %A to i64
; CHECK-NEXT:  #dbg_value(i64 [[C2]], {{.*}}, !DIExpression(),
; CHECK-NEXT:  #dbg_value(i64 [[C2]], {{.*}}, !DIExpression(),

; DIOP-DBGINFO-LABEL: @test_sext_zext(
; DIOP-DBGINFO-NEXT:  [[C2:%.*]] = zext i16 %A to i64
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 [[C2]], {{.*}}, !DIExpression(DIOpArg(0, i64), DIOpConvert(i32)),
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 [[C2]], {{.*}}, !DIExpression(DIOpArg(0, i64)),
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

; DIOP-DBGINFO-LABEL: @test_used_sext_zext(
; DIOP-DBGINFO-NEXT:  [[C1:%.*]] = zext i16 %A to i32
; DIOP-DBGINFO-NEXT:  #dbg_value(i32 [[C1]], {{.*}}, !DIExpression(DIOpArg(0, i32)),
; DIOP-DBGINFO-NEXT:  [[C2:%.*]] = zext i16 %A to i64
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 [[C2]], {{.*}}, !DIExpression(DIOpArg(0, i64)),
; DIOP-DBGINFO-NEXT:  call void @escape32(i32 %c1)
; DIOP-DBGINFO-NEXT:  ret i64 %c2
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

; DIOP-DBGINFO-LABEL: @test_cast_select(
; DIOP-DBGINFO-NEXT:  [[sel:%.*]] = select i1 %cond, i32 3, i32 5
; DIOP-DBGINFO-NEXT:  #dbg_value(i32 [[sel]], {{.*}}, !DIExpression(DIOpArg(0, i32), DIOpConvert(i16)),
; DIOP-DBGINFO-NEXT:  #dbg_value(i32 [[sel]], {{.*}}, !DIExpression(DIOpArg(0, i32)),
; DIOP-DBGINFO-NEXT:  ret i32 [[sel]]
  %sel = select i1 %cond, i16 3, i16 5
  %cast = zext i16 %sel to i32
  ret i32 %cast
}

define void @test_or(i64 %A) {
; CHECK-LABEL: @test_or(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 256, DW_OP_or, DW_OP_stack_value),

; FIXME: No way to represent bitwise or in DIOp-DIExpressions.
; DIOP-DBGINFO-LABEL: @test_or(
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 poison, {{.*}}, !DIExpression(DIOpArg(0, i64)),
  %1 = or i64 %A, 256
  ret void
}

define void @test_xor(i32 %A) {
; CHECK-LABEL: @test_xor(
; CHECK-NEXT:  #dbg_value(i32 %A, {{.*}}, !DIExpression(DW_OP_constu, 1, DW_OP_xor, DW_OP_stack_value),

; FIXME: No way to represent bitwise xor in DIOp-DIExpressions.
; DIOP-DBGINFO-LABEL: @test_xor(
; DIOP-DBGINFO-NEXT:  #dbg_value(i32 poison, {{.*}}, !DIExpression(DIOpArg(0, i32)),
  %1 = xor i32 %A, 1
  ret void
}

define void @test_sub_neg(i64 %A) {
; CHECK-LABEL: @test_sub_neg(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value),

; DIOP-DBGINFO-LABEL: @test_sub_neg(
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DIOpArg(0, i64), DIOpConstant(i64 -1), DIOpSub()),
  %1 = sub i64 %A, -1
  ret void
}

define void @test_sub_pos(i64 %A) {
; CHECK-LABEL: @test_sub_pos(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value),

; DIOP-DBGINFO-LABEL: @test_sub_pos(
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DIOpArg(0, i64), DIOpConstant(i64 1), DIOpSub()),
  %1 = sub i64 %A, 1
  ret void
}

define void @test_shl(i64 %A) {
; CHECK-LABEL: @test_shl(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 7, DW_OP_shl, DW_OP_stack_value),

; DIOP-DBGINFO-LABEL: @test_shl(
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DIOpArg(0, i64), DIOpConstant(i64 7), DIOpShl()),
  %1 = shl i64 %A, 7
  ret void
}

define void @test_lshr(i64 %A) {
; CHECK-LABEL: @test_lshr(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 7, DW_OP_shr, DW_OP_stack_value),

; DIOP-DBGINFO-LABEL: @test_lshr(
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DIOpArg(0, i64), DIOpConstant(i64 7), DIOpLShr()),
  %1 = lshr i64 %A, 7
  ret void
}

define void @test_ashr(i64 %A) {
; CHECK-LABEL: @test_ashr(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 7, DW_OP_shra, DW_OP_stack_value),

; DIOP-DBGINFO-LABEL: @test_ashr(
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DIOpArg(0, i64), DIOpConstant(i64 7), DIOpAShr()),
  %1 = ashr i64 %A, 7
  ret void
}

define void @test_mul(i64 %A) {
; CHECK-LABEL: @test_mul(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 7, DW_OP_mul, DW_OP_stack_value),

; DIOP-DBGINFO-LABEL: @test_mul(
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DIOpArg(0, i64), DIOpConstant(i64 7), DIOpMul()),
  %1 = mul i64 %A, 7
  ret void
}

define void @test_sdiv(i64 %A) {
; CHECK-LABEL: @test_sdiv(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 7, DW_OP_div, DW_OP_stack_value),

; DIOP-DBGINFO-LABEL: @test_sdiv(
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DIOpArg(0, i64), DIOpConstant(i64 7), DIOpDiv()),
  %1 = sdiv i64 %A, 7
  ret void
}

define void @test_srem(i64 %A) {
; CHECK-LABEL: @test_srem(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 7, DW_OP_mod, DW_OP_stack_value),

; FIXME: No way to represent srem in DIOp-DIExpressions.
; DIOP-DBGINFO-LABEL: @test_srem(
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 poison, {{.*}}, !DIExpression(DIOpArg(0, i64)),
  %1 = srem i64 %A, 7
  ret void
}

define void @test_ptrtoint(ptr %P) {
; CHECK-LABEL: @test_ptrtoint
; CHECK-NEXT:  #dbg_value(ptr %P, {{.*}}, !DIExpression(),

; DIOP-DBGINFO-LABEL: @test_ptrtoint
; DIOP-DBGINFO-NEXT:  #dbg_value(ptr %P, {{.*}}, !DIExpression(DIOpArg(0, ptr), DIOpReinterpret(i64)),
  %1 = ptrtoint ptr %P to i64
  ret void
}

define void @test_and(i64 %A) {
; CHECK-LABEL: @test_and(
; CHECK-NEXT:  #dbg_value(i64 %A, {{.*}}, !DIExpression(DW_OP_constu, 256, DW_OP_and, DW_OP_stack_value),

; FIXME: No way to represent bitwise and in DIOp-DIExpressions.
; DIOP-DBGINFO-LABEL: @test_and(
; DIOP-DBGINFO-NEXT:  #dbg_value(i64 poison, {{.*}}, !DIExpression(DIOpArg(0, i64)),
  %1 = and i64 %A, 256
  ret void
}

%struct.G = type { [4 x i16] }
%struct.S = type { i32, [10 x %struct.G] }

define void @test_gep(ptr %A) {
; CHECK-LABEL: @test_gep(
; CHECK-NEXT:  #dbg_value(ptr %A, {{.*}}, !DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value),

; DIOP-DBGINFO-LABEL: @test_gep(
; DIOP-DBGINFO-NEXT:  #dbg_value(ptr %A, {{.*}}, !DIExpression(DIOpArg(0, ptr), DIOpReinterpret(i64), DIOpConstant(i64 4), DIOpAdd(), DIOpReinterpret(ptr)),
  %1 = getelementptr %struct.S, ptr %A, i32 0, i32 1
  ret void
}

define void @test_gep_var_offset(ptr %A, i64 %B, i64 %C) {
; CHECK-LABEL: @test_gep_var_offset(
; CHECK-NEXT:  #dbg_value(!DIArgList(ptr %A, i64 %B, i64 %C), {{.*}}, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 8, DW_OP_mul, DW_OP_plus, DW_OP_LLVM_arg, 2, DW_OP_constu, 2, DW_OP_mul, DW_OP_plus, DW_OP_plus_uconst, 88, DW_OP_stack_value),

; DIOP-DBGINFO-LABEL: @test_gep_var_offset(
; DIOP-DBGINFO-NEXT:  #dbg_value(!DIArgList(ptr %A, i64 %B, i64 %C), {{.*}}, !DIExpression(DIOpArg(0, ptr), DIOpReinterpret(i64), DIOpArg(1, i64), DIOpConstant(i64 8), DIOpMul(), DIOpAdd(), DIOpArg(2, i64), DIOpConstant(i64 2), DIOpMul(), DIOpAdd(), DIOpConstant(i64 88), DIOpAdd(), DIOpReinterpret(ptr)),

  ; This is the following expression in infix: i64(A) + B*8 + C*2 + 88
  %1 = getelementptr %struct.S, ptr %A, i32 1, i32 1, i64 %B, i32 0, i64 %C
  ret void
}
