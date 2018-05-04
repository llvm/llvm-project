; RUN: llc -mtriple=x86_64-apple-darwin %s -o - -filetype=obj | \
; RUN:     llvm-dwarfdump --name i --name indirect - | FileCheck %s
;
; This is a hand-crafted testcase generated from:
;   int i = 23;
;   int *indirect = &i;

; CHECK: DW_TAG_variable
; CHECK:   DW_AT_name	("i")
; CHECK:   DW_AT_location	(DW_OP_addr 0x8, DW_OP_deref)
; CHECK: DW_TAG_variable
; CHECK:   DW_AT_name	("indirect")
; CHECK:   DW_AT_location	(DW_OP_addr 0x8

source_filename = "global-deref.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

@i = global i32 23, align 4
@indirect = global i32* @i, align 8, !dbg !6, !dbg !0, !dbg !14, !dbg !15, !dbg !16

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12, !13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression(DW_OP_deref))
!1 = distinct !DIGlobalVariable(name: "i", scope: !2, file: !3, line: 1, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "global-deref.c", directory: "/")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "indirect", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"PIC Level", i32 2}
; This is malformed, but too expensive to detect in the verifier.
!14 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!15 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression(DW_OP_LLVM_fragment, 0, 1))
!16 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression(DW_OP_deref, DW_OP_constu, 1, DW_OP_plus, DW_OP_stack_value))
