; RUN: llc -filetype=obj < %s -o %t 2>&1 | FileCheck --allow-empty --implicit-check-not='warning:' %s
; RUN: llvm-dwarfdump -verify %t
; RUN: llvm-dwarfdump %t | FileCheck %s --check-prefix=DWARF --implicit-check-not=DW_TAG

; DWARF:      DW_TAG_compile_unit

; Abstract subprogram.
; DWARF: [[FOO:.*]]:   DW_TAG_subprogram
; DWARF:               DW_AT_name ("foo"
; DWARF:               DW_AT_inline (DW_INL_inlined)

; Concrete subprogram.
; DWARF:               DW_TAG_subprogram
; DWARF:                 DW_AT_abstract_origin ([[FOO]]

; Concrete subprogram.
; DWARF:               DW_TAG_subprogram
; DWARF:                 DW_AT_abstract_origin ([[FOO]]

; Check that when DISubprogram is attached to two empty functions (with no lexical scopes),
; they get different DW_TAG_subprograms.

; ModuleID = 'example'

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64--linux"

!0 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !{})
!1 = !DIFile(filename: "example.c", directory: "/")
!2 = !DISubroutineType(types: !4)
!4 = !{}

!3 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!30}

!30 = !{i32 1, !"Debug Info Version", i32 3}

define void @foo() !dbg !0 {
entry:
  ret void
}

define void @foo_clone() !dbg !0 {
entry:
  ret void
}
