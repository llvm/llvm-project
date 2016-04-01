; Check that DEBUG_VALUE comments come through on a variety of targets.

define i32 @main() nounwind ssp !dbg !0 {
entry:
; CHECK: DEBUG_VALUE
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !7, metadata !DIExpression()), !dbg !9
  ret i32 0, !dbg !10
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13}

!0 = distinct !DISubprogram(name: "main", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, file: !12, scope: !1, type: !3)
!1 = !DIFile(filename: "/tmp/x.c", directory: "/Users/manav")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 2.9 (trunk 120996)", isOptimized: false, emissionKind: FullDebug, file: !12, enums: !6, retainedTypes: !6, subprograms: !11)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !{}
!7 = !DILocalVariable(name: "i", line: 3, scope: !8, file: !1, type: !5)
!8 = distinct !DILexicalBlock(line: 2, column: 12, file: !12, scope: !0)
!9 = !DILocation(line: 3, column: 11, scope: !8)
!10 = !DILocation(line: 4, column: 2, scope: !8)
!11 = !{!0}
!12 = !DIFile(filename: "/tmp/x.c", directory: "/Users/manav")
!13 = !{i32 1, !"Debug Info Version", i32 3}
