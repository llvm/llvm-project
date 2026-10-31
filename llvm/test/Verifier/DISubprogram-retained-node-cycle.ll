; RUN: llvm-as -disable-output %s 2>&1 | FileCheck %s

; A retained node (DILocalVariable) whose scope is a cyclic DILexicalBlock
; chain that never reaches a DISubprogram. 

; CHECK: invalid retained nodes, retained node does not belong to subprogram
; CHECK: DIScope scope chain must not contain a cycle
; CHECK: distinct !DILexicalBlock(scope: !{{[0-9]+}}
; CHECK: DIScope scope chain must not contain a cycle
; CHECK: distinct !DILexicalBlock(scope: !{{[0-9]+}}
; CHECK: warning: ignoring invalid debug info

define void @f() !dbg !5 {
entry:
  ret void, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "retained-cycle.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!6 = !{null}
!7 = !{!8}
!8 = !DILocalVariable(name: "x", scope: !9, file: !1, line: 2, type: !13)
!9 = distinct !DILexicalBlock(scope: !10, file: !1, line: 2, column: 3)
!10 = distinct !DILexicalBlock(scope: !9, file: !1, line: 3, column: 5)
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 1, column: 1, scope: !5)
