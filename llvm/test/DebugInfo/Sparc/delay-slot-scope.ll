; RUN: llc -O2 -mtriple=sparcv9 %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -O2 -mtriple=sparcv9 -filetype=obj %s -o - \
; RUN:   | llvm-dwarfdump --debug-info --debug-line - | FileCheck %s

; The only instruction of the lexical block for q is the load, which the delay
; slot filler moves into the delay slot of the retl. It still needs its own
; line table entry and must keep the lexical block and q in the DWARF.

; int get(int *p) {
;   ...
;   {
;     int *q = p + 1;
;     return *q;
;   ...
; }

; ASM-LABEL: get:
; ASM:       .loc 1 10 2 prologue_end
; ASM-NEXT:  retl
; ASM-NEXT:  {{^}}.Ltmp{{[0-9]+}}:
; ASM-NEXT:  .loc 1 8 7
; ASM-NEXT:  ldsw [%o0+4], %o0

; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_name ("get")
; CHECK:        DW_TAG_lexical_block
; CHECK-NEXT:     DW_AT_low_pc (0x0000000000000004)
; CHECK-NEXT:     DW_AT_high_pc (0x0000000000000008)
; CHECK:          DW_TAG_variable
; CHECK-NEXT:       DW_AT_location (DW_OP_breg8 O0+4, DW_OP_stack_value)
; CHECK-NEXT:       DW_AT_name ("q")

; CHECK:      Address Line Column File
; CHECK:      0x0000000000000000 10 2 1 0 0 0 is_stmt prologue_end
; CHECK-NEXT: 0x0000000000000004 8 7 1 0 0 0 is_stmt

define signext i32 @get(ptr %p) !dbg !4 {
entry:
  %q = getelementptr inbounds i8, ptr %p, i64 4
    #dbg_value(ptr %q, !10, !DIExpression(), !12)
  %v = load i32, ptr %q, align 4, !dbg !12
  ret i32 %v, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "blk.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "get", scope: !1, file: !1, line: 6, type: !5, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !9)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !8}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!9 = !{!10}
!10 = !DILocalVariable(name: "q", scope: !11, file: !1, line: 8, type: !8)
!11 = distinct !DILexicalBlock(scope: !4, file: !1, line: 7, column: 3)
!12 = !DILocation(line: 8, column: 7, scope: !11)
!13 = !DILocation(line: 10, column: 2, scope: !4)
