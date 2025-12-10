; Check that composite type DIEs go to debug_types section.

; RUN: llc -generate-type-units -filetype=obj %s -o - | llvm-dwarfdump -debug-info -debug-types - | FileCheck %s

; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_class_type
; CHECK: DW_AT_signature ([[SIG_A:0x[0-9a-f]+]])
; CHECK: DW_TAG_subprogram
; CHECK: NULL
; CHECK: DW_TAG_subprogram
; CHECK: "_ZN1A6AppendEv"
; CHECK: DW_TAG_class_type
; CHECK: DW_AT_signature ([[SIG_LAMBDA:0x[0-9a-f]+]])
; CHECK: DW_TAG_variable
; CHECK: NULL
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_inlined_subroutine
; CHECK: NULL
; CHECK: NULL

; CHECK:      .debug_types contents:
; CHECK:      Type Unit: {{.*}} type_signature = [[SIG_A]]
; CHECK:      DW_TAG_class_type
; CHECK-NOT:    DW_TAG
; CHECK:        DW_AT_name ("A")
; CHECK:      Type Unit: {{.*}} type_signature = [[SIG_LAMBDA]]
; CHECK:      DW_TAG_class_type
; CHECK:      DW_TAG_class_type
; CHECK-NOT:    DW_TAG
; CHECK:        DW_AT_decl_line (7)

target triple = "aarch64-unknown-linux-gnu"

define void @_Z1f1A() !dbg !4 {
entry:
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, emissionKind: FullDebug, globals: !2)
!1 = !DIFile(filename: "<stdin>", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "f", linkageName: "_Z1f1A", scope: !5, file: !5, line: 14, type: !6, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!5 = !DIFile(filename: "repro.ii", directory: "")
!6 = distinct !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 8, column: 12, scope: !9, inlinedAt: !16)
!9 = distinct !DISubprogram(name: "Append", linkageName: "_ZN1A6AppendEv", scope: !10, file: !5, line: 6, type: !11, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !12, retainedNodes: !13)
!10 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !5, line: 3, size: 32, flags: DIFlagTypePassByValue, elements: !2, identifier: "_ZTS1A")
!11 = distinct !DISubroutineType(types: !7)
!12 = !DISubprogram(name: "Append", linkageName: "_ZN1A6AppendEv", scope: !10, file: !5, line: 6, type: !11, scopeLine: 6, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!13 = !{!14}
!14 = !DILocalVariable(name: "raw_append", scope: !9, file: !5, line: 7, type: !15)
!15 = distinct !DICompositeType(tag: DW_TAG_class_type, scope: !9, file: !5, line: 7, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !2, identifier: "_ZTSZN1A6AppendEvEUlvE_")
!16 = distinct !DILocation(line: 14, column: 15, scope: !4)
