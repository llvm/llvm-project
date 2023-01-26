;; Check whether fortran assumed size array is accepted
;; which has upperBound absent in DISubrange

; RUN: llc -mtriple=x86_64-unknown-linux-gnu %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump  %t.o | FileCheck %s
; RUN: rm -rf %t
; RUN: mkdir %t
; RUN: sed -e "s/Fortran90/Fortran18/" %s > %t/test.ll
; RUN: llc -mtriple=x86_64-unknown-linux-gnu %t/test.ll -filetype=obj -o %t/test.o
; RUN: llvm-dwarfdump  %t/test.o | FileCheck %s

; CHECK-LABEL: DW_TAG_formal_parameter
; CHECK: DW_AT_name    ("array1")
; CHECK: DW_AT_type    ([[type1:0x[0-9a-f]+]]
; CHECK-LABEL: DW_TAG_formal_parameter
; CHECK: DW_AT_name    ("array2")
; CHECK: DW_AT_type    ([[type2:0x[0-9a-f]+]]
; CHECK: [[type1]]:   DW_TAG_array_type
; CHECK: DW_TAG_subrange_type
; CHECK: [[type2]]:   DW_TAG_array_type
; CHECK: DW_TAG_subrange_type
; CHECK: DW_AT_lower_bound     (4)
; CHECK: DW_AT_upper_bound     (9)
; CHECK: DW_TAG_subrange_type
; CHECK: DW_AT_lower_bound     (10)
;
;
;; original fortran program
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;subroutine sub (array1, array2)
;;  integer :: array1 (*)
;;  integer :: array2 (4:9, 10:*)
;;
;;  array1(7:8) = 9
;;  array2(5, 10) = 10
;;end subroutine
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; ModuleID = 'assumed_size_array.ll'
source_filename = "assumed_size_array.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.C344_sub_ = internal constant i32 10
@.C345_sub_ = internal constant i64 10
@.C351_sub_ = internal constant i64 5
@.C341_sub_ = internal constant i32 9
@.C322_sub_ = internal constant i64 1
@.C350_sub_ = internal constant i64 8
@.C349_sub_ = internal constant i64 7

define void @sub_(ptr noalias %array1, ptr noalias %array2) #0 !dbg !5 {
L.entry:
  %.dY0001_361 = alloca i64, align 8
  %"i$a_357" = alloca i64, align 8
  call void @llvm.dbg.declare(metadata ptr %array1, metadata !16, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.declare(metadata ptr %array2, metadata !18, metadata !DIExpression()), !dbg !17
  br label %L.LB1_364

L.LB1_364:                                        ; preds = %L.entry
  store i64 2, ptr %.dY0001_361, align 8, !dbg !19
  call void @llvm.dbg.declare(metadata ptr %"i$a_357", metadata !20, metadata !DIExpression()), !dbg !17
  store i64 7, ptr %"i$a_357", align 8, !dbg !19
  br label %L.LB1_359

L.LB1_359:                                        ; preds = %L.LB1_359, %L.LB1_364
  %0 = load i64, ptr %"i$a_357", align 8, !dbg !19
  call void @llvm.dbg.value(metadata i64 %0, metadata !22, metadata !DIExpression()), !dbg !17
  %1 = getelementptr i8, ptr %array1, i64 -4, !dbg !19
  %2 = getelementptr i32, ptr %1, i64 %0, !dbg !19
  store i32 9, ptr %2, align 4, !dbg !19
  %3 = load i64, ptr %"i$a_357", align 8, !dbg !19
  call void @llvm.dbg.value(metadata i64 %3, metadata !23, metadata !DIExpression()), !dbg !17
  %4 = add nsw i64 %3, 1, !dbg !19
  store i64 %4, ptr %"i$a_357", align 8, !dbg !19
  %5 = load i64, ptr %.dY0001_361, align 8, !dbg !19
  %6 = sub nsw i64 %5, 1, !dbg !19
  store i64 %6, ptr %.dY0001_361, align 8, !dbg !19
  %7 = load i64, ptr %.dY0001_361, align 8, !dbg !19
  %8 = icmp sgt i64 %7, 0, !dbg !19
  br i1 %8, label %L.LB1_359, label %L.LB1_383, !dbg !19

L.LB1_383:                                        ; preds = %L.LB1_359
  %9 = getelementptr i8, ptr %array2, i64 4, !dbg !24
  store i32 10, ptr %9, align 4, !dbg !24
  ret void, !dbg !25
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
!3 = !DIFile(filename: "assumed_size_array.f90", directory: "/tmp")
!4 = !{}
!5 = distinct !DISubprogram(name: "sub", scope: !2, file: !3, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !12}
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, align: 32, elements: !10)
!9 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DISubrange(lowerBound: 1)
!12 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, align: 32, elements: !13)
!13 = !{!14, !15}
!14 = !DISubrange(lowerBound: 4, upperBound: 9)
!15 = !DISubrange(lowerBound: 10)
!16 = !DILocalVariable(name: "array1", arg: 1, scope: !5, file: !3, line: 1, type: !8)
!17 = !DILocation(line: 0, scope: !5)
!18 = !DILocalVariable(name: "array2", arg: 2, scope: !5, file: !3, line: 1, type: !12)
!19 = !DILocation(line: 5, column: 1, scope: !5)
!20 = distinct !DILocalVariable(scope: !5, file: !3, type: !21, flags: DIFlagArtificial)
!21 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!22 = distinct !DILocalVariable(scope: !5, file: !3, type: !21, flags: DIFlagArtificial)
!23 = distinct !DILocalVariable(scope: !5, file: !3, type: !21, flags: DIFlagArtificial)
!24 = !DILocation(line: 6, column: 1, scope: !5)
!25 = !DILocation(line: 7, column: 1, scope: !5)
