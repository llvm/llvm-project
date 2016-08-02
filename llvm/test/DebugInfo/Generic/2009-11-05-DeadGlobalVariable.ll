; RUN: llc %s -o /dev/null
; Here variable bar is optimized away. Do not trip over while trying to generate debug info.


define i32 @foo() nounwind uwtable readnone ssp !dbg !5 {
entry:
  ret i32 42, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 139632)", isOptimized: true, emissionKind: 0, file: !17, enums: !1, retainedTypes: !1, subprograms: !3, globals: !12)
!1 = !{}
!3 = !{!5}
!5 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, file: !17, scope: !6, type: !7)
!6 = !DIFile(filename: "fb.c", directory: "/private/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!14}
!14 = !DIGlobalVariable(name: "bar", line: 2, isLocal: true, isDefinition: true, scope: !5, file: !6, type: !9)
!15 = !DILocation(line: 3, column: 3, scope: !16)
!16 = distinct !DILexicalBlock(line: 1, column: 11, file: !17, scope: !5)
!17 = !DIFile(filename: "fb.c", directory: "/private/tmp")
!18 = !{i32 1, !"Debug Info Version", i32 3}
