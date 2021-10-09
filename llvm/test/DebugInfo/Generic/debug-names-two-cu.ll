; XFAIL: -aix
; RUN: %llc_dwarf -accel-tables=Dwarf -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck %s
; RUN: llvm-dwarfdump -debug-names -verify %t | FileCheck --check-prefix=VERIFY %s

; Check the header
; CHECK: CU count: 2
; CHECK: Local TU count: 0
; CHECK: Foreign TU count: 0
; CHECK: Name count: 2
; CHECK: CU[0]: 0x{{[0-9a-f]*}}
; CHECK: CU[1]: 0x{{[0-9a-f]*}}

; CHECK: Abbreviation [[ABBREV:0x[0-9a-f]*]]
; CHECK-NEXT: Tag: DW_TAG_variable
; CHECK-NEXT: DW_IDX_compile_unit: DW_FORM_data1
; CHECK-NEXT: DW_IDX_die_offset: DW_FORM_ref4

; CHECK: String: 0x{{[0-9a-f]*}} "foobar2"
; CHECK-NEXT: Entry
; CHECK-NEXT: Abbrev: [[ABBREV]]
; CHECK-NEXT: Tag: DW_TAG_variable
; CHECK-NEXT: DW_IDX_compile_unit: 0x01
; CHECK-NEXT: DW_IDX_die_offset: 0x{{[0-9a-f]*}}

; CHECK: String: 0x{{[0-9a-f]*}} "foobar1"
; CHECK-NEXT: Entry
; CHECK-NEXT: Abbrev: [[ABBREV]]
; CHECK-NEXT: Tag: DW_TAG_variable
; CHECK-NEXT: DW_IDX_compile_unit: 0x00
; CHECK-NEXT: DW_IDX_die_offset: 0x{{[0-9a-f]*}}

; VERIFY: No errors.

!llvm.dbg.cu = !{!12, !22}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!0}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!0 = !{!"clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)"}
!4 = !{}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!3 = !DIFile(filename: "/tmp/cu2.c", directory: "/tmp")

@foobar1 = common dso_local global i8* null, align 8, !dbg !10
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "foobar1", scope: !12, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!12 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !15)
!15 = !{!10}

@foobar2 = common dso_local global i8* null, align 8, !dbg !20
!20 = !DIGlobalVariableExpression(var: !21, expr: !DIExpression())
!21 = distinct !DIGlobalVariable(name: "foobar2", scope: !22, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!22 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 325496) (llvm/trunk 325732)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !25)
!25 = !{!20}
