; REQUIRES: object-emission
; RUN: %llc_dwarf %s -O0 -filetype=obj -o %t.o
; RUN: llvm-dwarfdump %t.o --debug-info --verbose | FileCheck %s --implicit-check-not "{{DW_TAG|NULL}}"
; RUN: llvm-dwarfdump %t.o --debug-info --verify

; Test that we can inline from a different CU in a way that has triggered a bug.

define void @foo() !dbg !24 {
  ret void, !dbg !28
}

define void @bar() !dbg !34 {
  ret void, !dbg !35
}

!llvm.dbg.cu = !{!0, !3}
!llvm.module.flags = !{!10, !11}

!0 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !1, isOptimized: true, runtimeVersion: 5, emissionKind: FullDebug, globals: !2, imports: !2)
!1 = !DIFile(filename: "A.swift", directory: "")
!2 = !{}
!3 = distinct !DICompileUnit(language: DW_LANG_Swift, file: !4, isOptimized: true, runtimeVersion: 5, emissionKind: FullDebug, imports: !2)
!4 = !DIFile(filename: "B.swift", directory: "")
!10 = !{i32 7, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!24 = distinct !DISubprogram(name: "foo", scope: !25, file: !1, line: 116, type: !27, scopeLine: 116, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!25 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "FooTy", scope: !26, file: !1, size: 64, elements: !2)
!26 = !DIModule(scope: null, name: "Mod")
!27 = !DISubroutineType(types: !2)
!28 = !DILocation(line: 0, scope: !29, inlinedAt: !32)
!29 = distinct !DISubprogram(name: "init", scope: !31, file: !30, type: !27, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!30 = !DIFile(filename: "<compiler-generated>", directory: "")
!31 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "ModController", scope: !26, file: !4, size: 64, elements: !2, runtimeLang: DW_LANG_Swift)
!32 = !DILocation(line: 117, column: 29, scope: !33)
!33 = distinct !DILexicalBlock(scope: !24, file: !1, line: 116, column: 68)
!34 = distinct !DISubprogram(name: "bar", scope: !31, file: !4, line: 21, type: !27, scopeLine: 21, unit: !3, retainedNodes: !2)
!35 = !DILocation(line: 0, scope: !36, inlinedAt: !37)
!36 = distinct !DISubprogram(name: "goo", scope: !26, file: !30, type: !27, unit: !3, retainedNodes: !2)
!37 = !DILocation(line: 21, column: 26, scope: !34)

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_module
; CHECK:     DW_TAG_structure_type
; CHECK: [[INIT:0x.*]]: DW_TAG_subprogram
; CHECK;         DW_AT_name [DW_FORM_strp]  ({{.*}} = "init")
; CHECK:       DW_TAG_subprogram
; CHECK;         DW_AT_name [DW_FORM_strp]  ({{.*}} = "bar")
; CHECK:         DW_TAG_inlined_subroutine
; CHECK:           DW_AT_abstract_origin [DW_FORM_ref_addr] (0x00000000[[GOO:.*]] "goo")
; CHECK:         NULL
; CHECK:       NULL
; CHECK:     DW_TAG_structure_type
; CHECK:       DW_TAG_subprogram
; CHECK;         DW_AT_name [DW_FORM_strp]  ({{.*}} = "foo")
; CHECK:         DW_TAG_inlined_subroutine
; CHECK:           DW_AT_abstract_origin [DW_FORM_ref4]  ({{.*}} => {[[INIT]]} "init")
; CHECK:         NULL
; CHECK:       NULL
; CHECK:     NULL
; CHECK:   NULL
; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_module
; CHECK: 0x[[GOO]]: DW_TAG_subprogram
; CHECK;       DW_AT_name [DW_FORM_strp]  ({{.*}} = "goo")
; CHECK:     NULL
; CHECK:   NULL
