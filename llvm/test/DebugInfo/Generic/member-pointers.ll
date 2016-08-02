; REQUIRES: object-emission
; XFAIL: hexagon

; RUN: %llc_dwarf -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: DW_TAG_ptr_to_member_type
; CHECK: DW_TAG_ptr_to_member_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE:0x[0-9a-f]+]]})
; CHECK: [[TYPE]]:   DW_TAG_subroutine_type
; CHECK: DW_TAG_formal_parameter
; CHECK-NEXT: DW_AT_type
; CHECK-NEXT: DW_AT_artificial [DW_FORM_flag
; IR generated from clang -g with the following source:
; struct S {
; };
;
; int S::*x = 0;
; void (S::*y)(int) = 0;

@x = global i64 -1, align 8
@y = global { i64, i64 } zeroinitializer, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 ", isOptimized: false, emissionKind: 0, file: !15, enums: !1, retainedTypes: !1, subprograms: !1, globals: !3, imports:  !1)
!1 = !{}
!3 = !{!5, !10}
!5 = !DIGlobalVariable(name: "x", line: 4, isLocal: false, isDefinition: true, scope: null, file: !6, type: !7, variable: i64* @x)
!6 = !DIFile(filename: "simple.cpp", directory: "/home/blaikie/Development/scratch")
!7 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !8, extraData: !9)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", line: 1, size: 8, align: 8, file: !15, elements: !1)
!10 = !DIGlobalVariable(name: "y", line: 5, isLocal: false, isDefinition: true, scope: null, file: !6, type: !11, variable: { i64, i64 }* @y)
!11 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !12, extraData: !9)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14, !8}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer, baseType: !9)
!15 = !DIFile(filename: "simple.cpp", directory: "/home/blaikie/Development/scratch")
!16 = !{i32 1, !"Debug Info Version", i32 3}
