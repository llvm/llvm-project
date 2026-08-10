;; This test checks dataLocation field of DICompositeType

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

;; Test whether DW_AT_data_location is generated.
; CHECK:  !DICompositeType(tag: DW_TAG_array_type, baseType: !{{[0-9]+}}, size: 32, align: 32, elements: !{{[0-9]+}}, dataLocation: !{{[0-9]+}})
; CHECK:  !DICompositeType(tag: DW_TAG_array_type, baseType: !{{[0-9]+}}, size: 32, align: 32, elements: !{{[0-9]+}}, dataLocation: !DIExpression(DW_OP_constu, 3412))

; ModuleID = 'dataLocation.f90'
source_filename = "/dir/dataLocation.ll"

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !4, imports: !4)
!3 = !DIFile(filename: "fortsubrange.f90", directory: "/dir")
!4 = !{}
!5 = !{!6, !16}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32, align: 32, elements: !8, dataLocation: !10)
!7 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DISubrange(count: 19, lowerBound: 2)
!10 = distinct !DILocalVariable(scope: !11, file: !3, type: !15, flags: DIFlagArtificial)
!11 = !DILexicalBlock(scope: !12, file: !3, line: 1, column: 1)
!12 = distinct !DISubprogram(name: "main", scope: !2, file: !3, line: 1, type: !13, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!13 = !DISubroutineType(cc: DW_CC_program, types: !14)
!14 = !{null}
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32, align: 32)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32, align: 32, elements: !8, dataLocation: !DIExpression(DW_OP_constu, 3412))
