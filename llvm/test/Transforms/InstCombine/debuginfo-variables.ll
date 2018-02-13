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
