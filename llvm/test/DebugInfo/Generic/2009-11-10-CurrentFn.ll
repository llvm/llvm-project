; RUN: llc < %s -o /dev/null

define void @bar(i32 %i) nounwind uwtable ssp !dbg !5 {
entry:
  tail call void (...) @foo() nounwind, !dbg !14
  ret void, !dbg !16
}

declare void @foo(...)

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 139632)", isOptimized: true, emissionKind: FullDebug, file: !17, enums: !1, retainedTypes: !1, globals: !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "bar", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, file: !17, scope: !6, type: !7, variables: !9)
!6 = !DIFile(filename: "cf.c", directory: "/private/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{!11}
!11 = !DILocalVariable(name: "i", line: 3, arg: 1, scope: !5, file: !17, type: !12)
!12 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 3, column: 14, scope: !5)
!14 = !DILocation(line: 4, column: 3, scope: !15)
!15 = distinct !DILexicalBlock(line: 3, column: 17, file: !17, scope: !5)
!16 = !DILocation(line: 5, column: 1, scope: !15)
!17 = !DIFile(filename: "cf.c", directory: "/private/tmp")
!18 = !{i32 1, !"Debug Info Version", i32 3}
