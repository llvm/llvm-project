; RUN: llvm-as -disable-output %s 2>&1 | FileCheck %s

; Reject a cycle in a DILocalScope chain instead of looping indefinitely while
; looking for the enclosing DISubprogram.

; CHECK: DIScope scope chain must not contain a cycle
; CHECK: distinct !DILexicalBlock(scope: ![[BLOCK2:[0-9]+]]
; CHECK: warning: ignoring invalid debug info

define void @f() !dbg !5 {
entry:
  ret void, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "scope-cycle.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!6 = !{null}
!9 = distinct !DILexicalBlock(scope: !10, file: !1, line: 2, column: 3)
!10 = distinct !DILexicalBlock(scope: !9, file: !1, line: 4, column: 5)
!11 = !DILocation(line: 2, column: 3, scope: !9)
