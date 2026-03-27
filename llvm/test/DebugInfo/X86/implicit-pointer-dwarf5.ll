; RUN: llc -mtriple=x86_64-linux-gnu -filetype=obj -o %t %s
; RUN: llvm-dwarfdump --debug-info %t | FileCheck %s

; Test DW_OP_LLVM_implicit_pointer lowering to DWARF 5, using
; DW_TAG_dwarf_procedure for the artificial DIEs.

; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_name ("foo")
; CHECK:      DW_TAG_formal_parameter
; CHECK:        DW_AT_location (DW_OP_implicit_pointer
; CHECK:        DW_AT_name ("p")

; CHECK:      DW_TAG_dwarf_procedure
; CHECK-NEXT:   DW_AT_location (DW_OP_reg5 RDI)

; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_name ("bar")
; CHECK:      DW_TAG_formal_parameter
; CHECK:        DW_AT_location (DW_OP_implicit_pointer
; CHECK:        DW_AT_name ("q")

; CHECK:      DW_TAG_dwarf_procedure
; CHECK-NEXT:   DW_AT_const_value (42)

; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_name ("baz")
; CHECK:      DW_TAG_formal_parameter
; CHECK:        DW_AT_location (DW_OP_implicit_pointer
; CHECK:        DW_AT_name ("r")

; CHECK:      DW_TAG_dwarf_procedure
; CHECK-NEXT:   DW_AT_const_value

; CHECK:      DW_TAG_subprogram
; CHECK:        DW_AT_name ("dedup")
; CHECK:      DW_TAG_formal_parameter
; CHECK:        DW_AT_location (DW_OP_implicit_pointer
; CHECK:        DW_AT_name ("s")
; CHECK:      DW_TAG_formal_parameter
; CHECK:        DW_AT_location (DW_OP_implicit_pointer
; CHECK:        DW_AT_name ("t")

; CHECK:      DW_TAG_dwarf_procedure
; CHECK-NEXT:   DW_AT_const_value (99)
; CHECK-NOT:  DW_AT_const_value (99)

define internal i32 @foo(i32 noundef %p.0.val) !dbg !7 {
entry:
    #dbg_value(i32 %p.0.val, !12, !DIExpression(DW_OP_LLVM_implicit_pointer), !14)
  %add = add nsw i32 %p.0.val, 5, !dbg !15
  ret i32 %add, !dbg !16
}

define internal i32 @bar() !dbg !17 {
entry:
    #dbg_value(i32 42, !20, !DIExpression(DW_OP_LLVM_implicit_pointer), !21)
  ret i32 47, !dbg !22
}

define internal float @baz() !dbg !23 {
entry:
    #dbg_value(float 0x400921CAC0000000, !26, !DIExpression(DW_OP_LLVM_implicit_pointer), !27)
  ret float 0x400921CAC0000000, !dbg !28
}

define internal i32 @dedup() !dbg !29 {
entry:
    #dbg_value(i32 99, !32, !DIExpression(DW_OP_LLVM_implicit_pointer), !34)
    #dbg_value(i32 99, !33, !DIExpression(DW_OP_LLVM_implicit_pointer), !34)
  ret i32 198, !dbg !35
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!6, !10}
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!11 = !{!12}
!12 = !DILocalVariable(name: "p", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!14 = !DILocation(line: 2, scope: !7)
!15 = !DILocation(line: 2, column: 20, scope: !7)
!16 = !DILocation(line: 2, column: 10, scope: !7)

!17 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 5, type: !18, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !19)
!18 = !DISubroutineType(types: !9)
!19 = !{!20}
!20 = !DILocalVariable(name: "q", arg: 1, scope: !17, file: !1, line: 5, type: !10)
!21 = !DILocation(line: 5, scope: !17)
!22 = !DILocation(line: 5, column: 10, scope: !17)

!23 = distinct !DISubprogram(name: "baz", scope: !1, file: !1, line: 8, type: !24, scopeLine: 8, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !25)
!24 = !DISubroutineType(types: !38)
!25 = !{!26}
!26 = !DILocalVariable(name: "r", arg: 1, scope: !23, file: !1, line: 8, type: !36)
!27 = !DILocation(line: 8, scope: !23)
!28 = !DILocation(line: 8, column: 10, scope: !23)

!29 = distinct !DISubprogram(name: "dedup", scope: !1, file: !1, line: 11, type: !30, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !31)
!30 = !DISubroutineType(types: !9)
!31 = !{!32, !33}
!32 = !DILocalVariable(name: "s", arg: 1, scope: !29, file: !1, line: 11, type: !10)
!33 = !DILocalVariable(name: "t", arg: 2, scope: !29, file: !1, line: 11, type: !10)
!34 = !DILocation(line: 11, scope: !29)
!35 = !DILocation(line: 11, column: 10, scope: !29)

!36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !37, size: 64)
!37 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!38 = !{!37, !36}