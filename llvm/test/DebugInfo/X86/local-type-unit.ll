; RUN: llc --generate-type-units -O0 -filetype=obj < %s | \
; RUN:   llvm-dwarfdump -debug-types - | \
; RUN:   FileCheck %s

; Check that emission of a type unit for a local type inside a lexical
; block does not cause a crash.

; Local types inside lexical blocks are currently emitted as immediate children
; of their corresponding subprograms.

; CHECK: DW_TAG_type_unit
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_TAG_structure_type
; CHECK:       DW_AT_name{{.*}}"Local"

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_Z1gv() !dbg !4 {
entry:
  ret void, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = distinct !DISubprogram(name: "g", linkageName: "_Z1gv", scope: !1, file: !1, line: 9, type: !5, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !7, keyInstructions: true)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{!8}
!8 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Local", scope: !9, file: !1, line: 3, size: 32, flags: DIFlagTypePassByValue, elements: !10, identifier: "_ZTSZ1fvE5Local")
!9 = distinct !DILexicalBlock(scope: !4, file: !1, line: 2, column: 3)
!10 = !{}
!11 = !DILocation(line: 9, column: 12, scope: !9)
