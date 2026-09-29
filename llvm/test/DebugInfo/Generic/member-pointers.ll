; RUN: %llc_dwarf -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump -v -debug-info %t | FileCheck %s
; CHECK: DW_TAG_ptr_to_member_type
; CHECK: DW_TAG_ptr_to_member_type
; CHECK-NEXT: DW_AT_type [DW_FORM_ref4]       (cu + {{.*}} => {[[TYPE:0x[0-9a-f]+]]}
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

source_filename = "test/DebugInfo/Generic/member-pointers.ll"

@x = global i64 -1, align 8, !dbg !0
@y = global { i64, i64 } zeroinitializer, align 8, !dbg !7

!llvm.dbg.cu = !{!13}
!llvm.module.flags = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: null, file: !2, line: 4, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "simple.cpp", directory: "/home/blaikie/Development/scratch")
!3 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !4, extraData: !5)
!4 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!5 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !2, line: 1, size: 8, align: 8, elements: !6)
!6 = !{}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = !DIGlobalVariable(name: "y", scope: null, file: !2, line: 5, type: !9, isLocal: false, isDefinition: true)
!9 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !10, extraData: !5)
!10 = !DISubroutineType(types: !11)
!11 = !{null, !12, !4}
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, align: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!13 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.3 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !6, retainedTypes: !6, globals: !14, imports: !6)
!14 = !{!0, !7}
!15 = !{i32 1, !"Debug Info Version", i32 3}

