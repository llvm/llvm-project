; RUN: llc -O2 -mtriple=sparcv9 %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -O2 -mtriple=sparcv9 -filetype=obj %s -o - \
; RUN:   | llvm-dwarfdump --debug-info - | FileCheck %s --check-prefix=DWARF

; A leaf procedure stays in the caller's register window, so its parameters
; live in %o registers, while a function that opens a window keeps them in %i.

; ASM-LABEL: leaf:
; ASM:       !DEBUG_VALUE: leaf:p <- $o0
; ASM-LABEL: nonleaf:
; ASM:       !DEBUG_VALUE: nonleaf:p <- $i0

; DWARF:      DW_AT_name ("leaf")
; DWARF:      DW_TAG_formal_parameter
; DWARF-NEXT:   DW_AT_location (DW_OP_reg8 O0)
; DWARF-NEXT:   DW_AT_name ("p")

define i64 @leaf(ptr %p) !dbg !10 {
    #dbg_value(ptr %p, !14, !DIExpression(), !15)
  %v = load i64, ptr %p, align 8, !dbg !16
  ret i64 %v, !dbg !16
}

declare void @ext()

define i64 @nonleaf(ptr %p) !dbg !20 {
    #dbg_value(ptr %p, !21, !DIExpression(), !22)
  call void @ext(), !dbg !23
  %v = load i64, ptr %p, align 8, !dbg !23
  ret i64 %v, !dbg !23
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "hand-written", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "leaf.c", directory: "/tmp")
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!9 = !DISubroutineType(types: !{!5, !6})
!10 = distinct !DISubprogram(name: "leaf", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !{})
!14 = !DILocalVariable(name: "p", arg: 1, scope: !10, file: !1, line: 1, type: !6)
!15 = !DILocation(line: 0, scope: !10)
!16 = !DILocation(line: 2, column: 3, scope: !10)
!20 = distinct !DISubprogram(name: "nonleaf", scope: !1, file: !1, line: 5, type: !9, scopeLine: 5, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !{})
!21 = !DILocalVariable(name: "p", arg: 1, scope: !20, file: !1, line: 5, type: !6)
!22 = !DILocation(line: 0, scope: !20)
!23 = !DILocation(line: 7, column: 3, scope: !20)
