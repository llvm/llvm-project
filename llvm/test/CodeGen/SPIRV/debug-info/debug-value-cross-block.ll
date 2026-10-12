; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --asm-verbose=0 --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A record in one block naming a value defined in another. SPIR-V requires an
; id defined in a function block to dominate a non-phi use, and requires a
; block to precede every block it dominates, so a dominating definition has
; also been printed already and can be named.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "v"
; CHECK-DAG: [[VAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[NAME]]
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression{{ *$}}

; CHECK: [[SUM:%[0-9]+]] = OpIAdd
; The record is in the successor block, after its OpLabel, naming the entry
; block's definition.
; CHECK: OpLabel
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugValue [[VAR]] [[SUM]] [[EXPR]]{{ *$}}

target triple = "spirv64-unknown-unknown"

define spir_func i32 @f(i32 %x) !dbg !5 {
entry:
  %sum = add i32 %x, %x, !dbg !10
  %cmp = icmp slt i32 %sum, 0, !dbg !10
  br i1 %cmp, label %then, label %exit, !dbg !10

then:
    #dbg_value(i32 %sum, !9, !DIExpression(), !10)
  br label %exit, !dbg !10

exit:
  ret i32 %sum, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-value-cross-block.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "v", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 3, column: 3, scope: !5)
