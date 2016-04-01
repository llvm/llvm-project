; RUN: llc -O0 -mtriple=x86_64-apple-darwin10 < %s - | FileCheck %s
; Radar 8286101
; CHECK: .file   {{[0-9]+}} "<stdin>"

define i32 @foo() nounwind ssp !dbg !0 {
entry:
  ret i32 42, !dbg !8
}

define i32 @bar() nounwind ssp !dbg !6 {
entry:
  ret i32 21, !dbg !10
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17}

!0 = distinct !DISubprogram(name: "foo", linkageName: "foo", line: 53, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !14, scope: !1, type: !3)
!1 = !DIFile(filename: "", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 2.9 (trunk 114084)", isOptimized: false, emissionKind: FullDebug, file: !15, enums: !16, retainedTypes: !16, subprograms: !13)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = distinct !DISubprogram(name: "bar", linkageName: "bar", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !15, scope: !7, type: !3)
!7 = !DIFile(filename: "bug.c", directory: "/private/tmp")
!8 = !DILocation(line: 53, column: 13, scope: !9)
!9 = distinct !DILexicalBlock(line: 53, column: 11, file: !14, scope: !0)
!10 = !DILocation(line: 4, column: 13, scope: !11)
!11 = distinct !DILexicalBlock(line: 4, column: 13, file: !15, scope: !12)
!12 = distinct !DILexicalBlock(line: 4, column: 11, file: !15, scope: !6)
!13 = !{!0, !6}
!14 = !DIFile(filename: "", directory: "/private/tmp")
!15 = !DIFile(filename: "bug.c", directory: "/private/tmp")
!16 = !{}
!17 = !{i32 1, !"Debug Info Version", i32 3}
