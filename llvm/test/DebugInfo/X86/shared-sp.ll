; RUN: llc -filetype=obj < %s -o %t 2>&1 | FileCheck --allow-empty --implicit-check-not='warning:' %s
; RUN: llvm-dwarfdump -verify %t
; RUN: llvm-dwarfdump %t | FileCheck %s --check-prefix=DWARF --implicit-check-not=DW_TAG

; Check that when DISubprogram is attached to two functions, DWARF is produced
; correctly.

; DWARF:      DW_TAG_compile_unit

; Abstract subprogram.
; DWARF: [[FOO:.*]]:   DW_TAG_subprogram
; DWARF:               DW_AT_name ("foo"
; DWARF:               DW_AT_inline (DW_INL_inlined)
; DWARF: [[A:.*]]:       DW_TAG_variable
; DWARF:                   DW_AT_name ("a"
; DWARF:                 DW_TAG_structure_type
; DWARF:                   DW_TAG_member
; DWARF: [[C:.*]]:       DW_TAG_variable
; DWARF:                   DW_AT_name ("c"

; DWARF:               DW_TAG_base_type

; Concrete subprogram.
; DWARF:               DW_TAG_subprogram
; DWARF:                 DW_AT_abstract_origin ([[FOO]]
; DWARF:                 DW_TAG_variable
; DWARF:                   DW_AT_abstract_origin ([[A]]
; DWARF:                 DW_TAG_variable
; DWARF:                   DW_AT_abstract_origin ([[C]]

; Concrete subprogram.
; DWARF:               DW_TAG_subprogram
; DWARF:                 DW_AT_abstract_origin ([[FOO]]
; DWARF:                 DW_TAG_variable
; DWARF:                   DW_AT_abstract_origin ([[A]]
; DWARF:                 DW_TAG_variable
; DWARF:                   DW_AT_abstract_origin ([[C]]

; ModuleID = 'shared-sp.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

!0 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, retainedNodes: !{})
!1 = !DIFile(filename: "example.c", directory: "/")
!2 = !DISubroutineType(types: !3)
!3 = !{!5}
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

; Local variable.
!10 = !DILocalVariable(name: "a", scope: !0, file: !1, line: 2, type: !5)

; DICompositeType local to foo.
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", scope: !0, file: !1, line: 2, size: 32, elements: !12)
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "m", scope: !11, file: !1, line: 2, baseType: !5, size: 32)

; Local variable of type struct bar, local to foo.
!14 = !DILocalVariable(name: "c", scope: !0, file: !1, line: 2, type: !11)

!101 = !DILocation(line: 2, column: 5, scope: !0)
!102 = !DILocation(line: 3, column: 1, scope: !0)
!103 = !DILocation(line: 2, column: 12, scope: !0)

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!30}

!30 = !{i32 2, !"Debug Info Version", i32 3}

define i32 @foo() !dbg !0 {
entry:
  ; Local variable 'a' debug info.
  %a.addr = alloca i32, align 4, !dbg !101
    #dbg_declare(ptr %a.addr, !10, !DIExpression(), !101)
  store i32 42, ptr %a.addr, align 4, !dbg !101

  ; Local variable 'c' (struct bar) debug info.
  %c.addr = alloca %struct.bar, align 4, !dbg !103
    #dbg_declare(ptr %c.addr, !14, !DIExpression(), !103)

  ret i32 42, !dbg !102
}

define i32 @foo_clone() !dbg !0 {
entry:
  ; Local variable 'a' debug info.
  %a.addr = alloca i32, align 4, !dbg !101
    #dbg_declare(ptr %a.addr, !10, !DIExpression(), !101)
  store i32 42, ptr %a.addr, align 4, !dbg !101

  ; Local variable 'c' (struct bar) debug info.
  %c.addr = alloca %struct.bar, align 4, !dbg !103
    #dbg_declare(ptr %c.addr, !14, !DIExpression(), !103)

  ret i32 42, !dbg !102
}

%struct.bar = type { i32 }
