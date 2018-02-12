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

; CHECK: !8 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !9)
; CHECK: !10 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 2, type: !11)
; CHECK: !12 = !DILocation(line: 2, column: 1, scope: !5)
; CHECK: !13 = !DILocation(line: 1, column: 1, scope: !5)

; CHECK: !17 = !DILocalVariable(name: "3", scope: !15, file: !1, line: 4, type: !11)
; CHECK: !18 = !DILocation(line: 4, column: 1, scope: !15)
