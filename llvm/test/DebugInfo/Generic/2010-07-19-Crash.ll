; RUN: llc -o /dev/null < %s
; PR7662
; Do not add variables to !11 because it is a declaration entry.

define i32 @bar() nounwind readnone ssp !dbg !0 {
entry:
  ret i32 42, !dbg !9
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15}
!llvm.dbg.lv.foo = !{!7}

!0 = distinct !DISubprogram(name: "bar", linkageName: "bar", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !2, file: !12, scope: !1, type: !3)
!1 = !DIFile(filename: "one.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang 2.8", isOptimized: true, emissionKind: FullDebug, file: !12, enums: !14, retainedTypes: !14)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = distinct !DISubprogram(name: "foo", linkageName: "foo", line: 7, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !2, file: !12, scope: !1, type: !3, declaration: !11)
!7 = !DILocalVariable(name: "one", line: 8, scope: !8, file: !1, type: !5)
!8 = distinct !DILexicalBlock(line: 7, column: 18, file: !12, scope: !6)
!9 = !DILocation(line: 4, column: 3, scope: !10)
!10 = distinct !DILexicalBlock(line: 3, column: 11, file: !12, scope: !0)
!11 = !DISubprogram(name: "foo", linkageName: "foo", line: 7, isLocal: true, isDefinition: false, virtualIndex: 6, isOptimized: true, file: !12, scope: !1, type: !3)
!12 = !DIFile(filename: "one.c", directory: "/private/tmp")
!14 = !{}
!15 = !{i32 1, !"Debug Info Version", i32 3}
