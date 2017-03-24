; RUN: llc -mtriple=x86_64-apple-darwin %s -o - -filetype=obj | \
; RUN:     llvm-dwarfdump --debug-dump=info - | FileCheck %s
;
; Test emitting debug info for fragmented global values.
; This is a handcrafted example of an SROAed global variable.
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.anon = type { i32, i32 }

; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}}"point"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_location [DW_FORM_exprloc]	(<0x16> 03 04 00 00 00 00 00 00 00 93 04 03 00 00 00 00 00 00 00 00 93 04 )
;     [0x0000000000000004], piece 0x00000004, [0x0000000000000000], piece 0x00000004
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}}"part_const"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_location [DW_FORM_exprloc]	(<0x10> 03 08 00 00 00 00 00 00 00 93 04 10 02 9f 93 04 )
;     [0x0000000000000008], piece 0x00000004, constu 0x00000002, stack-value, piece 0x00000004
; CHECK-NOT: DW_TAG
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name {{.*}}"full_const"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_location [DW_FORM_exprloc]	(<0xa> 10 01 9f 93 04 10 02 9f 93 04 )
;     constu 0x00000001, stack-value, piece 0x00000004, constu 0x00000002, stack-value, piece 0x00000004
; CHECK-NOT: DW_TAG
@point.y = global i32 2, align 4, !dbg !13
@point.x = global i32 1, align 4, !dbg !12

@part_const.x = global i32 1, align 4, !dbg !14

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!10, !11}

!0 = distinct !DIGlobalVariable(name: "point", scope: !1, file: !2, line: 1, type: !5, isLocal: false, isDefinition: true)
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4)
!2 = !DIFile(filename: "g.c", directory: "/")
!3 = !{}
!4 = !{!12, !13, !14, !15, !17, !18}
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !2, line: 1, size: 64, elements: !6)
!6 = !{!7, !9}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !5, file: !2, line: 1, baseType: !8, size: 32)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_member, name: "y", scope: !5, file: !2, line: 1, baseType: !8, size: 32, offset: 32)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !DIGlobalVariableExpression(var: !0,  expr: !DIExpression(DW_OP_LLVM_fragment,  0, 32))
!13 = !DIGlobalVariableExpression(var: !0,  expr: !DIExpression(DW_OP_LLVM_fragment, 32, 32))
!14 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression(DW_OP_LLVM_fragment,  0, 32))
!15 = !DIGlobalVariableExpression(var: !16, expr: !DIExpression(DW_OP_constu, 2,
                                             DW_OP_stack_value, DW_OP_LLVM_fragment, 32, 32))
!16 = distinct !DIGlobalVariable(name: "part_const", scope: !1, file: !2, line: 1, type: !5, isLocal: false, isDefinition: true)
!17 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression(DW_OP_constu, 1,
                                             DW_OP_stack_value, DW_OP_LLVM_fragment,  0, 32))
!18 = !DIGlobalVariableExpression(var: !19, expr: !DIExpression(DW_OP_constu, 2,
                                             DW_OP_stack_value, DW_OP_LLVM_fragment, 32, 32))
!19 = distinct !DIGlobalVariable(name: "full_const", scope: !1, file: !2, line: 1, type: !5, isLocal: false, isDefinition: true)
