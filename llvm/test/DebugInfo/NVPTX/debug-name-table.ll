; RUN: llc -mtriple=nvptx64-nvidia-cuda -mattr=+ptx75 < %s | FileCheck %s
; RUN: %if ptxas-isa-7.5 %{ llc -mtriple=nvptx64-nvidia-cuda -mattr=+ptx75 < %s | %ptxas-verify %}

; DICompileUnit without 'nameTableKind: None' results in
; debug_pubnames and debug_pubtypes sections in DWARF. These sections
; use labels and label expressions, and ptxas requires PTX v7.5 to
; support them.

; CHECK-LABEL: .section .debug_pubnames
; CHECK-NEXT: {
; CHECK-NEXT: .b32 $L__pubNames_end0-$L__pubNames_start0
; CHECK-NEXT: $L__pubNames_start0:
; CHECK:      $L__pubNames_end0:
; CHECK-NEXT: }

; CHECK-LABEL: .section .debug_pubtypes
; CHECK-NEXT: {
; CHECK-NEXT: .b32 $L__pubTypes_end0-$L__pubTypes_start0
; CHECK-NEXT: $L__pubTypes_start0:
; CHECK:      $L__pubTypes_end0:
; CHECK-NEXT: }

; Function Attrs: nounwind ssp uwtable
define i32 @foo() #0 !dbg !4 {
entry:
  ret i32 0
}

attributes #0 = { nounwind ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 ", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "debug-name-table.c", directory: "")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !0, scopeLine: 5, file: !1, scope: !5, type: !6)
!5 = !DIFile(filename: "debug-name-table.c", directory: "")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{i32 2, !"Dwarf Version", i32 2}
!12 = !{i32 1, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.5.0 "}
