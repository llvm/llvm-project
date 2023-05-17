; RUN: llc -O0 -relocation-model pic < %s -o /dev/null
; REQUIRES: default_triple
; PR7545

@.str = private constant [4 x i8] c"one\00", align 1 ; <ptr> [#uses=1]
@.str1 = private constant [4 x i8] c"two\00", align 1 ; <ptr> [#uses=1]
@C.9.2167 = internal constant [2 x ptr] [ptr @.str, ptr @.str1]
!llvm.dbg.cu = !{!39}
!38 = !DIFile(filename: "pbmsrch.c", directory: "/Users/grawp/LLVM/test-suite/MultiSource/Benchmarks/MiBench/office-stringsearch")
!39 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build 9999)", isOptimized: true, emissionKind: FullDebug, file: !109, enums: !108, retainedTypes: !108)
!46 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, file: !109, baseType: !47)
!47 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!97 = distinct !DISubprogram(name: "main", linkageName: "main", line: 73, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: false, unit: !39, scope: !39, type: !98)
!98 = !DISubroutineType(types: !99)
!99 = !{!100}
!100 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!101 = !{ptr @C.9.2167}
!102 = !DILocalVariable(name: "find_strings", line: 75, scope: !103, file: !38, type: !104)
!103 = distinct !DILexicalBlock(line: 73, column: 0, file: null, scope: !97)
!104 = !DICompositeType(tag: DW_TAG_array_type, size: 85312, align: 64, file: !109, baseType: !46, elements: !105)
!105 = !{!106}
!106 = !DISubrange(count: 1333)
!107 = !DILocation(line: 73, scope: !103)
!108 = !{i32 0}
!109 = !DIFile(filename: "pbmsrch.c", directory: "/Users/grawp/LLVM/test-suite/MultiSource/Benchmarks/MiBench/office-stringsearch")

define i32 @main() nounwind ssp {
bb.nph:
  tail call void @llvm.dbg.declare(metadata ptr @C.9.2167, metadata !102, metadata !DIExpression()), !dbg !107
  ret i32 0, !dbg !107
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone
