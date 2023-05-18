;; This test checks rank field of DICompositeType

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

;; Test whether rank is generated.
; CHECK:  !DICompositeType(tag: DW_TAG_array_type, baseType: !{{[0-9]+}}, size: 32, align: 32, elements: !{{[0-9]+}}, dataLocation: !{{[0-9]+}}, rank: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 8, DW_OP_deref))

; ModuleID = 'rank.f90'
source_filename = "/dir/rank.ll"

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !4, imports: !4)
!3 = !DIFile(filename: "arank.f90", directory: "/dir")
!4 = !{}
!5 = !{!6}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32, align: 32, elements: !8, dataLocation: !16, rank: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 8, DW_OP_deref))
!7 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!8 = !{!9, !10, !11, !12, !13, !14, !15}
!9 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 80, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 120, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 112, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!10 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 128, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 168, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 160, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!11 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 176, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 216, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 208, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!12 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 224, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 264, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 256, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!13 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 272, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 312, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 304, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!14 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 320, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 360, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 352, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!15 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 368, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 408, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 400, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!16 = distinct !DILocalVariable(scope: !17, file: !3, type: !24, flags: DIFlagArtificial)
!17 = distinct !DISubprogram(name: "sub1", scope: !2, file: !3, line: 1, type: !18, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !20, !25}
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32, align: 32, elements: !21)
!21 = !{!22}
!22 = !DISubrange(lowerBound: 1, upperBound: !23)
!23 = distinct !DILocalVariable(scope: !17, file: !3, type: !24, flags: DIFlagArtificial)
!24 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!25 = !DICompositeType(tag: DW_TAG_array_type, baseType: !24, size: 1024, align: 64, elements: !26)
!26 = !{!27}
!27 = !DISubrange(lowerBound: 1, upperBound: 16)
