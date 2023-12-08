; This test checks that the array index type has signed encoding when
; the source language is Fortran.

; RUN: llc -O0 -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump %t | FileCheck %s

; CHECK:      DW_TAG_subrange_type
; CHECK-NEXT:     DW_AT_type ([[INDEX_TYPE:0x[0-9a-f]+]] "__ARRAY_SIZE_TYPE__")
; CHECK-NEXT:     DW_AT_lower_bound (-2)
; CHECK-NEXT:     DW_AT_upper_bound (2)

; CHECK:      [[INDEX_TYPE]]: DW_TAG_base_type
; CHECK-NEXT:     DW_AT_name ("__ARRAY_SIZE_TYPE__")
; CHECK-NEXT:     DW_AT_byte_size (0x08)
; CHECK-NEXT:     DW_AT_encoding (DW_ATE_signed)

source_filename = "test/DebugInfo/X86/fortran-array-index-type.ll"
target triple = "x86_64-unknown-linux-gnu"

@"test_$ARRAY_1D" = internal global [5 x i32] zeroinitializer, align 16, !dbg !0

!llvm.module.flags = !{!13, !14}
!llvm.dbg.cu = !{!6}
!omp_offload.info = !{}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "array_1d", linkageName: "test_$ARRAY_1D", scope: !6, file: !3, line: 2, type: !9, isLocal: true, isDefinition: true)
!3 = !DIFile(filename: "test.f90", directory: "/tests")
!6 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, producer: "Fortran Compiler", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !7, splitDebugInlining: false, nameTableKind: None)
!7 = !{!0}
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, elements: !11)
!10 = !DIBasicType(name: "INTEGER*4", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(lowerBound: -2, upperBound: 2)
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 2, !"Dwarf Version", i32 4}
