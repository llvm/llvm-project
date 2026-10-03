; RUN: llvm-as -disable-output %s 2>&1 | FileCheck %s

; The DILexicalBlock's parent is a DIFile rather than a DISubprogram. The chain
; is acyclic, so it must be diagnosed instead of asserting in
; DILocalScope::getSubprogram(), which casts every parent to DILocalScope.

; CHECK: DILocalScope scope chain must terminate at a DISubprogram
; CHECK: distinct !DILexicalBlock(scope: ![[FILE:[0-9]+]]
; CHECK: invalid local scope
; CHECK: warning: ignoring invalid debug info

define void @f() !dbg !5 {
entry:
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "non-local-parent.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!6 = !{null}
!7 = distinct !DILexicalBlock(scope: !1, file: !1, line: 2, column: 3)
!8 = !DILocation(line: 2, column: 3, scope: !7)
