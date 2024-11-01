; RUN: llc %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump  %t.o | FileCheck %s

; Test whether DW_OP_over is processed properly.
; CHECK-LABEL:  DW_TAG_array_type
; CHECK-LABEL:  DW_TAG_array_type
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_data_location
; CHECK:     DW_AT_allocated
; CHECK: DW_TAG_subrange_type
; CHECK:  DW_AT_lower_bound     (DW_OP_push_object_address, DW_OP_over, DW_OP_constu 0x30, DW_OP_mul, DW_OP_plus_uconst 0x50, DW_OP_plus, DW_OP_deref)
; CHECK:  DW_AT_upper_bound     (DW_OP_push_object_address, DW_OP_over, DW_OP_constu 0x30, DW_OP_mul, DW_OP_plus_uconst 0x78, DW_OP_plus, DW_OP_deref)
; CHECK:  DW_AT_byte_stride     (DW_OP_push_object_address, DW_OP_over, DW_OP_constu 0x30, DW_OP_mul, DW_OP_plus_uconst 0x70, DW_OP_plus, DW_OP_deref, DW_OP_lit4, DW_OP_mul)
; CHECK: DW_TAG_subrange_type
; CHECK:  DW_AT_lower_bound	(DW_OP_push_object_address, DW_OP_plus_uconst 0x80, DW_OP_deref)
; CHECK:  DW_AT_upper_bound	(DW_OP_push_object_address, DW_OP_plus_uconst 0xa8, DW_OP_deref)
; CHECK:  DW_AT_byte_stride	(DW_OP_push_object_address, DW_OP_plus_uconst 0xa0, DW_OP_deref, DW_OP_lit4, DW_OP_mul)
; CHECK: DW_TAG_subrange_type
; CHECK:  DW_AT_lower_bound	(DW_OP_push_object_address, DW_OP_plus_uconst 0xb0, DW_OP_deref)
; CHECK:  DW_AT_upper_bound	(DW_OP_push_object_address, DW_OP_plus_uconst 0xd8, DW_OP_deref)
; CHECK:  DW_AT_byte_stride	(DW_OP_push_object_address, DW_OP_plus_uconst 0xd0, DW_OP_deref, DW_OP_lit4, DW_OP_mul)

; Test case is hand written with the help of below Fortran source program.
; Generated IR is meaning less and goal of it is just to check the
; processing of DWARF operator DW_OP_over.
;------------------------------
;program main
;integer, allocatable :: arr(:,:,:)
;allocate(arr(2:20,3:30,4:40))
;arr(2,3,4)=99
;print *, arr
;end program main
;------------------------------

; ModuleID = 'over.ll'
source_filename = "allocated.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @MAIN_() !dbg !5 {
L.entry:
  %.Z0655_362 = alloca ptr, align 8
  %"arr$sd1_379" = alloca [28 x i64], align 8
  call void @llvm.dbg.declare(metadata ptr %.Z0655_362, metadata !8, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.declare(metadata ptr %.Z0655_362, metadata !12, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.declare(metadata ptr %"arr$sd1_379", metadata !14, metadata !DIExpression()), !dbg !11
  call void @llvm.dbg.declare(metadata ptr %"arr$sd1_379", metadata !19, metadata !DIExpression()), !dbg !11
  ret void, !dbg !25
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "over.f90", directory: "/dir")
!4 = !{}
!5 = distinct !DISubprogram(name: "main", scope: !2, file: !3, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = distinct !DILocalVariable(scope: !5, file: !3, type: !9, flags: DIFlagArtificial)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 64)
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 0, scope: !5)
!12 = distinct !DILocalVariable(scope: !5, file: !3, type: !13, flags: DIFlagArtificial)
!13 = !DIBasicType(name: "logical", size: 32, align: 32, encoding: DW_ATE_boolean)
!14 = distinct !DILocalVariable(name: "descriptor", scope: !5, file: !3, type: !15)
!15 = !DICompositeType(tag: DW_TAG_array_type, baseType: !16, size: 1792, align: 64, elements: !17)
!16 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DISubrange(lowerBound: 1, upperBound: 28)
!19 = !DILocalVariable(name: "arr", scope: !5, file: !3, type: !20)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 32, align: 32, elements: !21, dataLocation: !8, allocated: !12)
!21 = !{!22, !23, !24}
!22 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 80, DW_OP_plus, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 120, DW_OP_plus, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_over, DW_OP_constu, 48, DW_OP_mul, DW_OP_plus_uconst, 112, DW_OP_plus, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!23 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 128, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 168, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 160, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!24 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 176, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 216, DW_OP_deref), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 208, DW_OP_deref, DW_OP_constu, 4, DW_OP_mul))
!25 = !DILocation(line: 6, column: 1, scope: !5)
