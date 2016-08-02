; RUN: llc -mtriple=x86_64-apple-darwin -O0 -filetype=obj -o %t < %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

%class.A = type { [42 x i32] }

@a = global %class.A zeroinitializer, align 4

; Check that we can handle non-default array bounds. In this case, the array
; goes from [-3, 38].

; CHECK: DW_TAG_class_type
; CHECK: DW_TAG_member
; CHECK-NEXT:                   DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{[0-9a-f]*}}] = "x")
; CHECK-NEXT:                   DW_AT_type [DW_FORM_ref4]       (cu + 0x{{[0-9a-f]*}} => {[[ARRAY:0x[0-9a-f]*]]})

; CHECK: [[ARRAY]]: DW_TAG_array_type [{{.*}}] *
; CHECK-NEXT:                 DW_AT_type [DW_FORM_ref4]    (cu + 0x{{[0-9a-f]*}} => {[[BASE:0x[0-9a-f]*]]})

; CHECK: DW_TAG_subrange_type
; CHECK-NEXT:                   DW_AT_type [DW_FORM_ref4]  (cu + 0x{{[0-9a-f]*}} => {[[BASE2:0x[0-9a-f]*]]})
; CHECK-NEXT:                   DW_AT_lower_bound [DW_FORM_data8]       (0xfffffffffffffffd)
; CHECK-NEXT:                   DW_AT_count [DW_FORM_data1]       (0x2a)

; CHECK: [[BASE]]: DW_TAG_base_type
; CHECK: [[BASE2]]: DW_TAG_base_type
; CHECK-NEXT:                 DW_AT_name [DW_FORM_strp]       ( .debug_str[0x{{[0-9a-f]*}}] = "sizetype")
; CHECK-NEXT:                 DW_AT_byte_size [DW_FORM_data1] (0x08)
; CHECK-NEXT:                 DW_AT_encoding [DW_FORM_data1]  (DW_ATE_unsigned)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 (trunk 169136)", isOptimized: false, emissionKind: 0, file: !20, enums: !1, retainedTypes: !1, subprograms: !1, globals: !3, imports:  !1)
!1 = !{}
!3 = !{!5}
!5 = !DIGlobalVariable(name: "a", line: 1, isLocal: false, isDefinition: true, scope: null, file: !6, type: !7, variable: %class.A* @a)
!6 = !DIFile(filename: "t.cpp", directory: "/Volumes/Sandbox/llvm")
!7 = !DICompositeType(tag: DW_TAG_class_type, name: "A", line: 1, align: 32, file: !20, elements: !8)
!8 = !{!9, !14}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "x", line: 1, flags: DIFlagPrivate, file: !20, scope: !7, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_array_type, align: 32, baseType: !11, elements: !12)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DISubrange(count: 42, lowerBound: -3)
!14 = !DISubprogram(name: "A", line: 1, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, scopeLine: 1, file: !6, scope: !7, type: !15)
!15 = !DISubroutineType(types: !16)
!16 = !{null, !17}
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !7)
!20 = !DIFile(filename: "t.cpp", directory: "/Volumes/Sandbox/llvm")
!21 = !{i32 1, !"Debug Info Version", i32 3}
