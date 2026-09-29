;; This test checks DISubrange bounds

; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s

;; Test whether bounds are generated correctly.
; CHECK: !{{[0-9]+}} = !DISubrange(lowerBound: 3, upperBound: ![[NODE:[0-9]+]], stride: !DIExpression(DW_OP_constu, 4))
; CHECK: ![[NODE]] = distinct !DILocalVariable


; ModuleID = 'fortsubrange.ll'
source_filename = "fortsubrange.ll"

define void @MAIN_() !dbg !5 {
L.entry:
  %.Z0640_333 = alloca i32*, align 8
  %"arr$sd1_349" = alloca [16 x i64], align 8
  call void @llvm.dbg.declare(metadata i32** %.Z0640_333, metadata !8, metadata !DIExpression(DW_OP_deref)), !dbg !15
  call void @llvm.dbg.declare(metadata [16 x i64]* %"arr$sd1_349", metadata !13, metadata !DIExpression(DW_OP_plus_uconst, 120)), !dbg !15
  ret void, !dbg !16
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "fortsubrange.f90", directory: "/dir")
!4 = !{}
!5 = distinct !DISubprogram(name: "main", scope: !2, file: !3, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "arr", scope: !5, file: !3, type: !9)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 32, align: 32, elements: !11)
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(lowerBound: 3, upperBound: !13, stride: !DIExpression(DW_OP_constu, 4))
!13 = distinct !DILocalVariable(scope: !5, file: !3, type: !14, flags: DIFlagArtificial)
!14 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!15 = !DILocation(line: 0, scope: !5)
!16 = !DILocation(line: 6, column: 1, scope: !5)
