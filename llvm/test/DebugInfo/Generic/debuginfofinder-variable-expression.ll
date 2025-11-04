; RUN: opt -passes='print<module-debuginfo>' -disable-output 2>&1 < %s \
; RUN:   | FileCheck %s

; This is to track DebugInfoFinder's ability to find the debug info metadata,
; in particular, properly visit DIVariableExpression.

; CHECK: Compile unit: DW_LANG_C_plus_plus from /somewhere/source.cpp
; CHECK: Subprogram: foo from /somewhere/source.cpp:1 ('_Z3foov')
; CHECK: Type: DW_TAG_subroutine_type
; CHECK: Type: T from /somewhere/source.cpp:2 DW_TAG_structure_type
; CHECK: Type: int_bit_size DW_ATE_signed
; CHECK: Type: x from /somewhere/source.cpp:3 DW_TAG_member
; CHECK: Type: array_type DW_TAG_array_type
; CHECK: Type: int DW_ATE_signed
; CHECK: Type: sr__int_range DW_TAG_subrange_type
; CHECK: Type: int_lower_bound DW_ATE_signed
; CHECK: Type: int_upper_bound DW_ATE_signed
; CHECK: Type: int_stride DW_ATE_signed
; CHECK: Type: int_derived_size DW_ATE_signed
; CHECK: Type: int_derived_offset DW_ATE_signed

; ModuleID = '<stdin>'
source_filename = "llvm/test/DebugInfo/Generic/debuginfofinder-variable-expression.ll"

define noundef i32 @_Z3foov() !dbg !7 {
entry:
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 21.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "source.cpp", directory: "/somewhere")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{!"clang version 21.0.0git"}
!7 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "v", scope: !7, file: !1, line: 8, type: !12)
!12 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "T", scope: !7, file: !1, line: 2, size: !13, flags: DIFlagTypePassByValue, elements: !17)
!13 = !DIVariableExpression(expr: !DIExpression(DW_OP_LLVM_arg, 0), vars: !14)
!14 = !{!15}
!15 = !DILocalVariable(name: "var_bit_size", scope: !7, file: !1, line: 8, type: !16)
!16 = !DIBasicType(name: "int_bit_size", size: 32, encoding: DW_ATE_signed)
!17 = !{!18}
!18 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !12, file: !1, line: 3, baseType: !27, size: !19, offset: !23)
!19 = !DIVariableExpression(expr: !DIExpression(DW_OP_LLVM_arg, 0), vars: !20)
!20 = !{!21}
!21 = !DILocalVariable(name: "var_derived_size", scope: !7, file: !1, line: 8, type: !22)
!22 = !DIBasicType(name: "int_derived_size", size: 32, encoding: DW_ATE_signed)
!23 = !DIVariableExpression(expr: !DIExpression(DW_OP_LLVM_arg, 0), vars: !24)
!24 = !{!25}
!25 = !DILocalVariable(name: "var_derived_offset", scope: !7, file: !1, line: 8, type: !26)
!26 = !DIBasicType(name: "int_derived_offset", size: 32, encoding: DW_ATE_signed)
!27 = !DICompositeType(tag: DW_TAG_array_type, name: "array_type", baseType: !28, size: 64, align: 32, elements: !29)
!28 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!29 = !{!30}
!30 = !DISubrangeType(name: "sr__int_range", size: 32, align: 32, baseType: !28, lowerBound: !31, upperBound: !35, stride: !39)
!31 = !DIVariableExpression(expr: !DIExpression(DW_OP_LLVM_arg, 0), vars: !32)
!32 = !{!33}
!33 = !DILocalVariable(name: "var_lower_bound", scope: !7, file: !1, line: 8, type: !34)
!34 = !DIBasicType(name: "int_lower_bound", size: 32, encoding: DW_ATE_signed)
!35 = !DIVariableExpression(expr: !DIExpression(DW_OP_LLVM_arg, 0), vars: !36)
!36 = !{!37}
!37 = !DILocalVariable(name: "var_upper_bound", scope: !7, file: !1, line: 8, type: !38)
!38 = !DIBasicType(name: "int_upper_bound", size: 32, encoding: DW_ATE_signed)
!39 = !DIVariableExpression(expr: !DIExpression(DW_OP_LLVM_arg, 0), vars: !40)
!40 = !{!41}
!41 = !DILocalVariable(name: "var_stride", scope: !7, file: !1, line: 8, type: !42)
!42 = !DIBasicType(name: "int_stride", size: 32, encoding: DW_ATE_signed)
