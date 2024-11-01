; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump - | FileCheck --implicit-check-not "{{DW_TAG|NULL}}" %s

; Test that retained unused (unreferenced) types emission.

; Compiled from
; $ clang -cc1 -debug-info-kind=unused-types test.cpp -emit-llvm

; void test_unused() {
;   struct Y {};
;   {
;     struct X {};
;   }
; }

; CHECK: DW_TAG_compile_unit
; CHECK:   DW_TAG_subprogram
; CHECK:     DW_AT_name	("test_unused")
; CHECK:     DW_TAG_structure_type
; CHECK:       DW_AT_name	("Y")

; FIXME: here should be DW_TAG_lexical_block as a parent of structure 'X'.
; But it's not possible to reliably emit a lexical block for which a LexicalScope
; wasn't created, so we just fallback to the most close parent DIE
; (see DwarfCompileUnit::getOrCreateLexicalBlockDIE() for details).

; CHECK:     DW_TAG_structure_type
; CHECK:       DW_AT_name	("X")
; CHECK:     NULL
; CHECK:   NULL

define dso_local void @_Z11test_unusedv() !dbg !5 {
entry:
  ret void, !dbg !16
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{!3, !10}
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Y", scope: !5, file: !4, line: 2, size: 8, flags: DIFlagTypePassByValue, elements: !8)
!4 = !DIFile(filename: "test.cpp", directory: "/")
!5 = distinct !DISubprogram(name: "test_unused", linkageName: "_Z11test_unusedv", scope: !4, file: !4, line: 1, type: !6, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{}
!9 = !{!3, !10}
!10 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X", scope: !11, file: !4, line: 4, size: 8, flags: DIFlagTypePassByValue, elements: !8)
!11 = distinct !DILexicalBlock(scope: !5, file: !4, line: 3, column: 3)
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{!"clang version 15.0.0"}
!16 = !DILocation(line: 6, column: 1, scope: !5)
