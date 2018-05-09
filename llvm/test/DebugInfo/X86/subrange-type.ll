; RUN: llc -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s

; Make sure that the base type from the subrange type has a name.
; CHECK: DW_TAG_subrange_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]     (cu + 0x{{[0-9a-f]+}} => {[[SUBTYPE:0x[0-9a-f]*]]}
; CHECK: [[SUBTYPE]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name

define i32 @main() nounwind uwtable !dbg !5 {
entry:
  %retval = alloca i32, align 4
  %i = alloca [2 x i32], align 4
  store i32 0, i32* %retval
  call void @llvm.dbg.declare(metadata [2 x i32]* %i, metadata !10, metadata !DIExpression()), !dbg !15
  ret i32 0, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!18}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.3 (trunk 171472) (llvm/trunk 171487)", isOptimized: false, emissionKind: FullDebug, file: !17, enums: !1, retainedTypes: !1, globals: !1, imports:  !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "main", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !6, scope: !6, type: !7, retainedNodes: !1)
!6 = !DIFile(filename: "foo.c", directory: "/usr/local/google/home/echristo/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "i", line: 4, scope: !11, file: !6, type: !12)
!11 = distinct !DILexicalBlock(line: 3, column: 0, file: !6, scope: !5)
!12 = !DICompositeType(tag: DW_TAG_array_type, size: 64, align: 32, baseType: !9, elements: !13)
!13 = !{!14}
!14 = !DISubrange(count: 2)
!15 = !DILocation(line: 4, scope: !11)
!16 = !DILocation(line: 6, scope: !11)
!17 = !DIFile(filename: "foo.c", directory: "/usr/local/google/home/echristo/tmp")
!18 = !{i32 1, !"Debug Info Version", i32 3}
