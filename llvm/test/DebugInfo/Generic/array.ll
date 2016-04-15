; RUN: %llc_dwarf -O0 < %s | FileCheck %s
; Do not emit AT_upper_bound for an unbounded array.
; radar 9241695
define i32 @main() nounwind ssp !dbg !0 {
entry:
  %retval = alloca i32, align 4
  %a = alloca [0 x i32], align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata [0 x i32]* %a, metadata !6, metadata !DIExpression()), !dbg !11
  ret i32 0, !dbg !12
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!16}

!0 = distinct !DISubprogram(name: "main", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !2, scopeLine: 3, file: !14, scope: !1, type: !3)
!1 = !DIFile(filename: "array.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 129138)", isOptimized: false, emissionKind: FullDebug, file: !14, enums: !15, retainedTypes: !15, imports:  null)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DILocalVariable(name: "a", line: 4, scope: !7, file: !1, type: !8)
!7 = distinct !DILexicalBlock(line: 3, column: 12, file: !14, scope: !0)
!8 = !DICompositeType(tag: DW_TAG_array_type, align: 32, file: !14, scope: !2, baseType: !5, elements: !9)
!9 = !{!10}
;CHECK: section_info:
;CHECK: DW_TAG_subrange_type
;CHECK-NEXT: DW_AT_type
;CHECK-NOT: DW_AT_lower_bound
;CHECK-NOT: DW_AT_upper_bound
;CHECK-NEXT: End Of Children Mark
!10 = !DISubrange(count: -1)
!11 = !DILocation(line: 4, column: 7, scope: !7)
!12 = !DILocation(line: 5, column: 3, scope: !7)
!14 = !DIFile(filename: "array.c", directory: "/private/tmp")
!15 = !{}
!16 = !{i32 1, !"Debug Info Version", i32 3}
