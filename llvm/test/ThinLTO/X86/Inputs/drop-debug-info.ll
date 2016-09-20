; This file is checked-in as a .bc file, because the debug info version is
; intentionally out-of-date and llvm-as will drop it before writing the bitcode

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"


@argc = global i8 0, align 1, !dbg !21

define void @globalfunc() {
entry:
  %0 = load i8, i8* @argc, align 1
  tail call void @llvm.dbg.value(metadata i8 %0, i64 0, metadata !19, metadata !29), !dbg !DILocation(scope: !13)
  ret void
}


declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, enums: !2, retainedTypes: !3, globals: !20, imports: !2, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_class_type, name: "C", line: 2, size: 8, align: 8, file: !1, elements: !5, identifier: "_ZTS1C")
!5 = !{!6}
!6 = !DISubprogram(name: "test", file: !1, scope: !4, type: !7, isDefinition: false)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !11, !11, !11, null}
!9 = !DIBasicType(encoding: DW_ATE_signed, size: 32, align: 32, name: "int")
!10 = !DIDerivedType(baseType: !4, tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!13 = distinct !DISubprogram(name: "test_with_debug", linkageName: "test_with_debug", line: 6, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 6, file: !1, scope: !14, type: !15, variables: !17)
!14 = !DIFile(filename: "test.cpp", directory: "")
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{!18, !19}
!18 = !DILocalVariable(name: "c", line: 7, scope: !13, file: !14, type: !4)
!19 = !DILocalVariable(name: "lc", line: 8, scope: !13, file: !14, type: !11)
!20 = !{!21}
!21 = !DIGlobalVariable(name: "argc", line: 1, isLocal: false, isDefinition: true, scope: null, file: !14, type: !11)
!22 = !{i32 2, !"Dwarf Version", i32 4}
!23 = !{i32 2, !"Debug Info Version", i32 0}
!25 = !DILocation(line: 8, column: 3, scope: !13)
!29 = !DIExpression()
