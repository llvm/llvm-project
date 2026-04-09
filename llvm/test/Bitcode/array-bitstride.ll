;; Test bit stride of arrays.

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

; CHECK:  !DICompositeType(tag: DW_TAG_array_type, baseType: !{{[0-9]+}}, size: 32, align: 32, elements: !{{[0-9]+}}, bitStride: i32 7)

; ModuleID = 'stride.adb'
source_filename = "/dir/stride.ll"

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Ada95, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !4, imports: !4)
!3 = !DIFile(filename: "stride.adb", directory: "/dir")
!4 = !{}
!5 = !{!6, !17}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32, align: 32, elements: !8, dataLocation: !10, associated: !15)
!7 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DISubrange(count: 19, lowerBound: 2)
!10 = distinct !DILocalVariable(scope: !11, file: !3, type: !14, flags: DIFlagArtificial)
!11 = distinct !DISubprogram(name: "main", scope: !2, file: !3, line: 1, type: !12, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!12 = !DISubroutineType(cc: DW_CC_program, types: !13)
!13 = !{null}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32, align: 32)
!15 = distinct !DILocalVariable(scope: !11, file: !3, type: !16, flags: DIFlagArtificial)
!16 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32, align: 32, elements: !8, bitStride: i32 7)
