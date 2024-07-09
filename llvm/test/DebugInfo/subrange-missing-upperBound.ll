; RUN: %llc_dwarf %s -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; REQUIRES: object-emission


; ModuleID = 'test.ll'
source_filename = "test.f90"

!llvm.module.flags = !{!1}
!llvm.dbg.cu = !{!2}

!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, emissionKind: FullDebug, retainedTypes: !5)
!3 = !DIFile(filename: "test.f90", directory: "dir")
!5 = !{!6, !10, !13}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, elements: !8)
!7 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DISubrange(lowerBound: 2, stride: 16)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, elements: !11)
!11 = !{!12}
!12 = !DISubrange()
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, elements: !14)
!14 = !{!15}
!15 = !DIGenericSubrange(lowerBound: !16, stride: !17)
!16 = !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 48, DW_OP_deref)
!17 = !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 56, DW_OP_deref)

; Test that debug info is generated correctly in the absence of 'count' and
; 'upperBound' in DISubrange/DIGenericSubrange.

; CHECK-LABEL: DW_TAG_subrange_type
; CHECK-NEXT:   DW_AT_type
; CHECK-NEXT:   DW_AT_lower_bound     (2)
; CHECK-NEXT:   DW_AT_byte_stride     (16)

; CHECK-LABEL: DW_TAG_subrange_type
; CHECK-NEXT:   DW_AT_type

; CHECK-LABEL: DW_TAG_generic_subrange
; CHECK-NEXT:   DW_AT_type
; CHECK-NEXT:   DW_AT_lower_bound     (DW_OP_push_object_address, DW_OP_plus_uconst 0x30, DW_OP_deref)
; CHECK-NEXT:   DW_AT_byte_stride     (DW_OP_push_object_address, DW_OP_plus_uconst 0x38, DW_OP_deref)
