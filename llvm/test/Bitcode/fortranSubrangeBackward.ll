;; This test checks Backward compatibility of DISubrange bounds
; REQUIRES: x86_64-linux

; RUN: llvm-dis -o - %s.bc | FileCheck %s

;; Test whether bounds are generated correctly.
; CHECK: !DISubrange(count: 15, lowerBound: 3)
; CHECK: !DISubrange(count: ![[NODE:[0-9]+]], lowerBound: 3)
; CHECK: ![[NODE]] = distinct !DILocalVariable


; ModuleID = 'fortsubrange.ll'
source_filename = "fortsubrange.ll"

define void @MAIN_() !dbg !10 {
L.entry:
  %.Z0640_333 = alloca i32*, align 8
  %"arr$sd1_349" = alloca [16 x i64], align 8
  call void @llvm.dbg.declare(metadata i32** %.Z0640_333, metadata !13, metadata !DIExpression(DW_OP_deref)), !dbg !19
  call void @llvm.dbg.declare(metadata [16 x i64]* %"arr$sd1_349", metadata !17, metadata !DIExpression(DW_OP_plus_uconst, 120)), !dbg !19
  ret void, !dbg !20
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !4, imports: !4)
!3 = !DIFile(filename: "fortsubrange.f90", directory: "/dir")
!4 = !{}
!5 = !{!6}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32, align: 32, elements: !8)
!7 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DISubrange(count: 15, lowerBound: 3)
!10 = distinct !DISubprogram(name: "main", scope: !2, file: !3, line: 1, type: !11, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!11 = !DISubroutineType(cc: DW_CC_program, types: !12)
!12 = !{null}
!13 = !DILocalVariable(name: "arr", scope: !10, file: !3, type: !14)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 32, align: 32, elements: !15)
!15 = !{!16}
!16 = !DISubrange(count: !17, lowerBound: 3)
!17 = distinct !DILocalVariable(scope: !10, file: !3, type: !18, flags: DIFlagArtificial)
!18 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!19 = !DILocation(line: 0, scope: !10)
!20 = !DILocation(line: 6, column: 1, scope: !10)
