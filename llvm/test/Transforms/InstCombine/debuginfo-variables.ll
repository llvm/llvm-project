; RUN: opt < %s -debugify -instcombine -S | FileCheck %s

define i64 @test_sext_zext(i16 %A) {
; CHECK-LABEL: @test_sext_zext(
; CHECK-NEXT:  [[C2:%.*]] = zext i16 %A to i64
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i64 [[C2]], metadata !8, metadata !DIExpression()), !dbg !13
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i64 [[C2]], metadata !10, metadata !DIExpression()), !dbg !12
  %c1 = zext i16 %A to i32
  %c2 = sext i32 %c1 to i64
  ret i64 %c2
}

define void @test_or(i64 %A) {
; CHECK-LABEL: @test_or(
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i64 %A, metadata !17, metadata !DIExpression(DW_OP_constu, 256, DW_OP_or, DW_OP_stack_value)), !dbg !18
  %1 = or i64 %A, 256
  ret void
}

define void @test_xor(i32 %A) {
; CHECK-LABEL: @test_xor(
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i32 %A, metadata !22, metadata !DIExpression(DW_OP_constu, 1, DW_OP_xor, DW_OP_stack_value)), !dbg !23
  %1 = xor i32 %A, 1
  ret void
}

define void @test_sub_neg(i64 %A) {
; CHECK-LABEL: @test_sub_neg(
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i64 %A, metadata !27, metadata !DIExpression(DW_OP_plus_uconst, 1, DW_OP_stack_value)), !dbg !28
  %1 = sub i64 %A, -1
  ret void
}

define void @test_sub_pos(i64 %A) {
; CHECK-LABEL: @test_sub_pos(
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i64 %A, metadata !32, metadata !DIExpression(DW_OP_constu, 1, DW_OP_minus, DW_OP_stack_value)), !dbg !33
  %1 = sub i64 %A, 1
  ret void
}

define void @test_shl(i64 %A) {
; CHECK-LABEL: @test_shl(
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i64 %A, metadata !37, metadata !DIExpression(DW_OP_constu, 7, DW_OP_shl, DW_OP_stack_value)), !dbg !38
  %1 = shl i64 %A, 7
  ret void
}

define void @test_lshr(i64 %A) {
; CHECK-LABEL: @test_lshr(
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i64 %A, metadata !42, metadata !DIExpression(DW_OP_constu, 7, DW_OP_shr, DW_OP_stack_value)), !dbg !43
  %1 = lshr i64 %A, 7
  ret void
}

define void @test_ashr(i64 %A) {
; CHECK-LABEL: @test_ashr(
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i64 %A, metadata !47, metadata !DIExpression(DW_OP_constu, 7, DW_OP_shra, DW_OP_stack_value)), !dbg !48
  %1 = ashr i64 %A, 7
  ret void
}

define void @test_mul(i64 %A) {
; CHECK-LABEL: @test_mul(
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i64 %A, metadata !52, metadata !DIExpression(DW_OP_constu, 7, DW_OP_mul, DW_OP_stack_value)), !dbg !53
  %1 = mul i64 %A, 7
  ret void
}

define void @test_sdiv(i64 %A) {
; CHECK-LABEL: @test_sdiv(
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i64 %A, metadata !57, metadata !DIExpression(DW_OP_constu, 7, DW_OP_div, DW_OP_stack_value)), !dbg !58
  %1 = sdiv i64 %A, 7
  ret void
}

define void @test_srem(i64 %A) {
; CHECK-LABEL: @test_srem(
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i64 %A, metadata !62, metadata !DIExpression(DW_OP_constu, 7, DW_OP_mod, DW_OP_stack_value)), !dbg !63
  %1 = srem i64 %A, 7
  ret void
}

; CHECK: !8 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !9)
; CHECK: !10 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 2, type: !11)
; CHECK: !12 = !DILocation(line: 2, column: 1, scope: !5)
; CHECK: !13 = !DILocation(line: 1, column: 1, scope: !5)

; CHECK: !17 = !DILocalVariable(name: "3", scope: !15, file: !1, line: 4, type: !11)
; CHECK: !18 = !DILocation(line: 4, column: 1, scope: !15)

; CHECK: !22 = !DILocalVariable(name: "4", scope: !20, file: !1, line: 6, type: !9)
; CHECK: !23 = !DILocation(line: 6, column: 1, scope: !20)

; CHECK: !27 = !DILocalVariable(name: "5", scope: !25, file: !1, line: 8, type: !11)
; CHECK: !28 = !DILocation(line: 8, column: 1, scope: !25)

; CHECK: !32 = !DILocalVariable(name: "6", scope: !30, file: !1, line: 10, type: !11)
; CHECK: !33 = !DILocation(line: 10, column: 1, scope: !30)

; CHECK: !37 = !DILocalVariable(name: "7", scope: !35, file: !1, line: 12, type: !11)
; CHECK: !38 = !DILocation(line: 12, column: 1, scope: !35)

; CHECK: !42 = !DILocalVariable(name: "8", scope: !40, file: !1, line: 14, type: !11)
; CHECK: !43 = !DILocation(line: 14, column: 1, scope: !40)

; CHECK: !47 = !DILocalVariable(name: "9", scope: !45, file: !1, line: 16, type: !11)
; CHECK: !48 = !DILocation(line: 16, column: 1, scope: !45)

; CHECK: !52 = !DILocalVariable(name: "10", scope: !50, file: !1, line: 18, type: !11)
; CHECK: !53 = !DILocation(line: 18, column: 1, scope: !50)

; CHECK: !57 = !DILocalVariable(name: "11", scope: !55, file: !1, line: 20, type: !11)
; CHECK: !58 = !DILocation(line: 20, column: 1, scope: !55)

; CHECK: !62 = !DILocalVariable(name: "12", scope: !60, file: !1, line: 22, type: !11)
; CHECK: !63 = !DILocation(line: 22, column: 1, scope: !60)
