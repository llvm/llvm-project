;; This test checks generation of DIGenericSubrange.

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

;; Test whether DIGenericSubrange is generated.
; CHECK: !DIGenericSubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 88, DW_OP_plus, DW_OP_deref), lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 80, DW_OP_plus, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 112, DW_OP_plus, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))

; ModuleID = 'generic_subrange.f90'
source_filename = "/dir/generic_subrange.ll"

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !4, imports: !4)
!3 = !DIFile(filename: "generic_subrange.f90", directory: "/dir")
!4 = !{}
!5 = !{!6}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32, align: 32, elements: !8, dataLocation: !10, rank: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 8, DW_OP_deref))
!7 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!8 = !{!9}
!9 = !DIGenericSubrange(count: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 88, DW_OP_plus, DW_OP_deref), lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 80, DW_OP_plus, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 112, DW_OP_plus, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!10 = distinct !DILocalVariable(scope: !11, file: !3, type: !18, flags: DIFlagArtificial)
!11 = distinct !DISubprogram(name: "sub1", scope: !2, file: !3, line: 1, type: !12, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14, !19}
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32, align: 32, elements: !15)
!15 = !{!16}
!16 = !DISubrange(lowerBound: 1, upperBound: !17)
!17 = distinct !DILocalVariable(scope: !11, file: !3, type: !18, flags: DIFlagArtificial)
!18 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 1024, align: 64, elements: !20)
!20 = !{!21}
!21 = !DISubrange(lowerBound: 1, upperBound: 16)
