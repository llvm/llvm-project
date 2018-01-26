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

; CHECK: !8 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !9)
; CHECK: !9 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
; CHECK: !10 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 2, type: !11)
; CHECK: !11 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
; CHECK: !12 = !DILocation(line: 2, column: 1, scope: !5)
; CHECK: !13 = !DILocation(line: 1, column: 1, scope: !5)
