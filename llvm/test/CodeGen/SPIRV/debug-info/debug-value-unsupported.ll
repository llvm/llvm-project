; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --implicit-check-not=DebugValue --implicit-check-not=DebugOperation --implicit-check-not="OpCapability Float64" --implicit-check-not="OpTypeFloat 64"
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Shapes this backend does not emit a DebugValue for. An unavailable value has
; no result id to name, and an expression with no NonSemantic counterpart has
; no emitted DebugExpression. A cv-qualified type is not emitted as a
; DebugType, so its variable gets no DebugLocalVariable. A rejected constant
; record must not leave behind an otherwise unused semantic type, constant or
; capability, whether its expression, type, or parent scope rejects it. A value
; defined in a sibling block does not dominate the record, and SPIR-V requires
; a definition in a function block to dominate a non-phi use.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[UNAVAILABLE:%[0-9]+]] = OpString "unavailable"
; CHECK-DAG: [[EXPRESSION:%[0-9]+]] = OpString "unsupported_expression"
; CHECK-DAG: [[CONSTANT:%[0-9]+]] = OpString "unsupported_constant"
; CHECK-DAG: OpString "qualified_constant"
; CHECK-DAG: OpString "scope_rejected_constant"
; CHECK-DAG: [[SIBLING:%[0-9]+]] = OpString "sibling_block_value"
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[UNAVAILABLE]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[EXPRESSION]]
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[CONSTANT]]
; The sibling-block variable still gets a DebugLocalVariable, which is what
; says its record reached the availability test rather than an earlier one.
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[SIBLING]]
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugExpression{{ *$}}
; CHECK: OpFunction

target triple = "spirv64-unknown-unknown"

define spir_func i32 @f(i32 %x) !dbg !5 {
entry:
    #dbg_value(i32 poison, !10, !DIExpression(), !12)
    #dbg_value(i32 %x, !11, !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_signed), !12)
    #dbg_value(double 1.000000e+00, !13, !DIExpression(DW_OP_LLVM_convert, 64, DW_ATE_unsigned), !12)
    #dbg_value(double 2.000000e+00, !15, !DIExpression(), !12)
  ret i32 %x, !dbg !12
}

; A qualifier in the subprogram signature prevents DebugTypeFunction and
; DebugFunction emission. The local variable therefore has no parent scope,
; so its constant must not change the semantic module requirements.
define spir_func i32 @missing_scope(i32 %x) !dbg !17 {
entry:
    #dbg_value(double 3.000000e+00, !20, !DIExpression(), !21)
  ret i32 %x, !dbg !21
}

; A value defined in a sibling block reaches the record through neither a
; dominating definition nor the module section, so the record is dropped.
define spir_func i32 @not_dominating(i32 %x) !dbg !23 {
entry:
  %c = icmp slt i32 %x, 0, !dbg !26
  br i1 %c, label %a, label %b, !dbg !26

a:
  %v = add i32 %x, 1, !dbg !26
  br label %join, !dbg !26

b:
    #dbg_value(i32 %v, !25, !DIExpression(), !26)
  br label %join, !dbg !26

join:
  %r = phi i32 [ %v, %a ], [ 0, %b ], !dbg !26
  ret i32 %r, !dbg !26
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-value-unsupported.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DILocalVariable(name: "unavailable", scope: !5, file: !1, line: 3, type: !7)
!11 = !DILocalVariable(name: "unsupported_expression", scope: !5, file: !1, line: 4, type: !7)
!12 = !DILocation(line: 5, column: 3, scope: !5)
!13 = !DILocalVariable(name: "unsupported_constant", scope: !5, file: !1, line: 5, type: !14)
!14 = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
!15 = !DILocalVariable(name: "qualified_constant", scope: !5, file: !1, line: 6, type: !16)
!16 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !14)
!17 = distinct !DISubprogram(name: "missing_scope", linkageName: "missing_scope", scope: !1, file: !1, line: 8, type: !18, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!18 = !DISubroutineType(types: !19)
!19 = !{!22, !7}
!20 = !DILocalVariable(name: "scope_rejected_constant", scope: !17, file: !1, line: 9, type: !14)
!21 = !DILocation(line: 10, column: 3, scope: !17)
!22 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!23 = distinct !DISubprogram(name: "not_dominating", linkageName: "not_dominating", scope: !1, file: !1, line: 12, type: !4, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!25 = !DILocalVariable(name: "sibling_block_value", scope: !23, file: !1, line: 13, type: !7)
!26 = !DILocation(line: 14, column: 3, scope: !23)
