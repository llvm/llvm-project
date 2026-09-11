; RUN: llvm-as -disable-output %s 2>&1 | FileCheck %s

; Cyclic DILexicalBlock used as a DILocalVariable scope related to a DbgVariableRecord.

; CHECK: DIScope scope chain must not contain a cycle
; CHECK: warning: ignoring invalid debug info

define void @f() !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata i32 0, metadata !12, metadata !DIExpression()), !dbg !13
  ret void, !dbg !13
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "dbg-record-scope-cycle.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!6 = !{null}
!9 = distinct !DILexicalBlock(scope: !10, file: !1, line: 2, column: 3)
!10 = distinct !DILexicalBlock(scope: !9, file: !1, line: 4, column: 5)
!12 = !DILocalVariable(name: "x", scope: !9, file: !1, line: 2, type: !14)
!13 = !DILocation(line: 2, column: 3, scope: !5)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
