; RUN: opt -S -passes=bdce < %s | FileCheck %s

; Check that BDCE preserves the signed-extension semantics of direct debug uses
; when replacing a sext with a zext because the sign bits are not demanded by
; program uses.

define i32 @test(i32 %a) !dbg !5 {
; CHECK-LABEL: define i32 @test(
; CHECK-SAME: i32 [[A:%.*]])
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ZEXT:%.*]] = zext i32 [[A]] to i64
; CHECK-NEXT:    #dbg_value(i32 [[A]], [[VAR:![0-9]+]], !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_LLVM_convert, 64, DW_ATE_signed, DW_OP_stack_value)
; CHECK-NEXT:    [[OR:%.*]] = or i64 [[ZEXT]], 5
; CHECK-NEXT:    [[TRUNC:%.*]] = trunc i64 [[OR]] to i32
; CHECK-NEXT:    ret i32 [[TRUNC]]
entry:
  %sext = sext i32 %a to i64
    #dbg_value(i64 %sext, !9, !DIExpression(), !10)
  %or = or i64 %sext, 5
  %trunc = trunc i64 %or to i32
  ret i32 %trunc
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{!11, !11}
!8 = !{!9}
!9 = !DILocalVariable(name: "extended", scope: !5, file: !1, line: 2, type: !12)
!10 = !DILocation(line: 2, column: 1, scope: !5)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
