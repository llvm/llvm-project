; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Exercise DebugFunctionDefinition placement across multiple defined functions
; that call each other, with and without function-local OpVariable instructions.
; Each function's definition must reference its own OpFunction id, placed after
; the entry OpLabel (no locals) or after the last function-local OpVariable.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[NAME_LEAF:%[0-9]+]] = OpString "leaf_no_vars"
; CHECK-DAG: [[NAME_WITH:%[0-9]+]] = OpString "helper_with_vars"
; CHECK-DAG: [[NAME_CALLER:%[0-9]+]] = OpString "caller_no_vars"
; CHECK-DAG: [[NAME_ORCH:%[0-9]+]] = OpString "orchestrator"
; CHECK-DAG: [[DF_LEAF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME_LEAF]]
; CHECK-DAG: [[DF_WITH:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME_WITH]]
; CHECK-DAG: [[DF_CALLER:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME_CALLER]]
; CHECK-DAG: [[DF_ORCH:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME_ORCH]]
; CHECK-DAG: OpDecorate [[LEAF:%[0-9]+]] LinkageAttributes "leaf_no_vars" Export
; CHECK-DAG: OpDecorate [[WITH:%[0-9]+]] LinkageAttributes "helper_with_vars" Export
; CHECK-DAG: OpDecorate [[CALLER:%[0-9]+]] LinkageAttributes "caller_no_vars" Export
; CHECK-DAG: OpDecorate [[ORCH:%[0-9]+]] LinkageAttributes "orchestrator" Export

; leaf_no_vars: no local variables -> DebugFunctionDefinition after OpLabel.
; CHECK: [[LEAF]] = OpFunction %{{.*}} ; -- Begin function leaf_no_vars
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF_LEAF]] [[LEAF]]
; CHECK-NEXT: OpReturnValue
; CHECK-NEXT: OpFunctionEnd

; helper_with_vars: two local variables -> DebugFunctionDefinition after them.
; CHECK: [[WITH]] = OpFunction %{{.*}} ; -- Begin function helper_with_vars
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpVariable {{.*}} Function
; CHECK-NEXT: OpVariable {{.*}} Function
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF_WITH]] [[WITH]]
; CHECK: OpReturnValue
; CHECK-NEXT: OpFunctionEnd

; caller_no_vars: calls both helpers, no locals -> after OpLabel.
; CHECK: [[CALLER]] = OpFunction %{{.*}} ; -- Begin function caller_no_vars
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF_CALLER]] [[CALLER]]
; CHECK-NEXT: OpFunctionCall
; CHECK-NEXT: OpFunctionCall
; CHECK-NEXT: OpReturnValue
; CHECK-NEXT: OpFunctionEnd

; orchestrator: no args, no locals, only calls -> after OpLabel.
; CHECK: [[ORCH]] = OpFunction %{{.*}} ; -- Begin function orchestrator
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF_ORCH]] [[ORCH]]
; CHECK-NEXT: OpFunctionCall
; CHECK-NEXT: OpFunctionCall
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @leaf_no_vars(i32 %value) !dbg !5 {
entry:
  ret i32 %value, !dbg !8
}

define spir_func i32 @helper_with_vars(i32 %value) !dbg !9 {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 %value, ptr %x
  store i32 0, ptr %y
  %tmp = call spir_func i32 @leaf_no_vars(i32 %value), !dbg !12
  %sum = add i32 %tmp, 1
  ret i32 %sum, !dbg !13
}

define spir_func i32 @caller_no_vars(i32 %value) !dbg !14 {
entry:
  %a = call spir_func i32 @helper_with_vars(i32 %value), !dbg !17
  %b = call spir_func i32 @leaf_no_vars(i32 %a), !dbg !18
  ret i32 %b, !dbg !19
}

define spir_func void @orchestrator() !dbg !20 {
entry:
  %unused1 = call spir_func i32 @leaf_no_vars(i32 0), !dbg !23
  %unused2 = call spir_func i32 @helper_with_vars(i32 1), !dbg !24
  ret void, !dbg !25
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-function-definition-calls.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "leaf_no_vars", linkageName: "leaf_no_vars", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 3, column: 3, scope: !5)

!9 = distinct !DISubprogram(name: "helper_with_vars", linkageName: "helper_with_vars", scope: !1, file: !1, line: 5, type: !4, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DILocalVariable(name: "x", scope: !9, file: !1, line: 6, type: !7)
!11 = !DILocalVariable(name: "y", scope: !9, file: !1, line: 7, type: !7)
!12 = !DILocation(line: 10, column: 10, scope: !9)
!13 = !DILocation(line: 12, column: 3, scope: !9)

!14 = distinct !DISubprogram(name: "caller_no_vars", linkageName: "caller_no_vars", scope: !1, file: !1, line: 14, type: !4, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!17 = !DILocation(line: 16, column: 8, scope: !14)
!18 = !DILocation(line: 17, column: 8, scope: !14)
!19 = !DILocation(line: 18, column: 3, scope: !14)

!20 = distinct !DISubprogram(name: "orchestrator", linkageName: "orchestrator", scope: !1, file: !1, line: 20, type: !21, scopeLine: 20, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!21 = !DISubroutineType(types: !22)
!22 = !{null}
!23 = !DILocation(line: 22, column: 13, scope: !20)
!24 = !DILocation(line: 23, column: 13, scope: !20)
!25 = !DILocation(line: 24, column: 3, scope: !20)
