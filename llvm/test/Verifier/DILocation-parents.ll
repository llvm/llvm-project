; RUN: llvm-as -disable-output %s -o - 2>&1 | FileCheck %s
source_filename = "t.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

define void @f() !dbg !4 {
entry:
; CHECK: scope points into the type hierarchy
; CHECK: DILocation
  ret void, !dbg !7
}

define void @g() !dbg !8 {
entry:
; CHECK: scope points into the type hierarchy
; CHECK: DILexicalBlockFile
  ret void, !dbg !9
}

; CHECK: warning: ignoring invalid debug info

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !5, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, unit: !0)
!5 = !DISubroutineType(types: !{})
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !5, isDefinition: false, scopeLine: 2)
!7 = !DILocation(line: 2, scope: !6)
!8 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 2, type: !5, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, unit: !0)
!9 = !DILocation(line: 2, scope: !10, inlinedAt: !11)
!10 = !DILexicalBlockFile(scope: !6, file: !2, discriminator: 0)
!11 = !DILocation(line: 2, scope: !8)
