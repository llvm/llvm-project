;; This test checks whether DW_AT_rank attribute accepts DIExpression.

; RUN: llc %s -mtriple=x86_64 -filetype=obj -o %t.o
; RUN: llvm-dwarfdump  %t.o | FileCheck %s

;; Test whether DW_AT_data_location is generated.
; CHECK-LABEL:  DW_TAG_array_type

; CHECK:        DW_AT_rank (7)

;; Test case is hand written with the help of below testcase
;;------------------------------
;;subroutine sub(arank)
;;  real :: arank(..)
;;  print *, RANK(arank)
;;end
;;------------------------------

; ModuleID = 'dwarfdump-rank.ll'
source_filename = "dwarfdump-rank.ll"

define void @sub_(i64* noalias %arank, i64* noalias %"arank$sd") !dbg !5 {
L.entry:
  call void @llvm.dbg.value(metadata i64* %arank, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64* %"arank$sd", metadata !19, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64* %"arank$sd", metadata !29, metadata !DIExpression()), !dbg !18
  ret void, !dbg !18
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "rank.f90", directory: "/dir")
!4 = !{}
!5 = distinct !DISubprogram(name: "sub", scope: !2, file: !3, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !14}
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !10)
!9 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!10 = !{!11}
!11 = !DISubrange(lowerBound: 1, upperBound: !12)
!12 = distinct !DILocalVariable(scope: !5, file: !3, type: !13, flags: DIFlagArtificial)
!13 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 1024, align: 64, elements: !15)
!15 = !{!16}
!16 = !DISubrange(lowerBound: 1, upperBound: 16)
!17 = distinct !DILocalVariable(scope: !5, file: !3, type: !13, flags: DIFlagArtificial)
!18 = !DILocation(line: 0, scope: !5)
!19 = !DILocalVariable(name: "arank", arg: 1, scope: !5, file: !3, line: 1, type: !20)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !21, dataLocation: !17, rank: 7)
!21 = !{!22, !23, !24, !25, !26, !27, !28}
!22 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 80, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 120, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 112, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!23 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 128, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 168, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 160, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!24 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 176, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 216, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 208, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!25 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 224, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 264, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 256, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!26 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 272, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 312, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 304, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!27 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 320, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 360, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 352, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!28 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 368, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 408, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 400, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!29 = !DILocalVariable(arg: 2, scope: !5, file: !3, line: 1, type: !14, flags: DIFlagArtificial)
