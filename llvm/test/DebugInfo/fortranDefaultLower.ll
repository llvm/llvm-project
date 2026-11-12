;; This test checks whether fortran default lowerBound is removed.
; REQUIRES: x86_64-linux

; RUN: %llc_dwarf %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s

;; fortran default DW_AT_lower_bound(1) is not dumped.
; CHECK-LABEL: DW_TAG_subrange_type
; CHECK-NEXT:   DW_AT_type
; CHECK-NEXT:   DW_AT_upper_bound     (5)

;; fortran non-default lowerBound=2 is dumped.
; CHECK-LABEL: DW_TAG_subrange_type
; CHECK-NEXT:   DW_AT_type
; CHECK-NEXT:   DW_AT_lower_bound     (2)
; CHECK-NEXT:   DW_AT_upper_bound     (6)

; ModuleID = 'fortranDefaultLower.ll'
source_filename = "fortranDefaultLower.f90"

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !4, imports: !4)
!3 = !DIFile(filename: "fortranDefaultLower.f90", directory: "dir")
!4 = !{}
!5 = !{!6, !10}
!6 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 160, align: 32, elements: !8)
!7 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DISubrange(lowerBound: 1, upperBound: 5)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 160, align: 32, elements: !11)
!11 = !{!12}
!12 = !DISubrange(lowerBound: 2, upperBound: 6)
