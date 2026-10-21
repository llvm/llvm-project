; Verify that SPIRVNonSemanticDebugHandler emits OpTypeVoid when the module
; has debug info but no void-returning function. Every NSDI OpExtInst uses
; OpTypeVoid as its result-type operand; without it the output is invalid
; SPIR-V. The spirv-val run checks for that.

; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: [[ext_inst:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK: OpExtInst [[type_void]] [[ext_inst]] DebugCompilationUnit

define spir_func i32 @get_value() !dbg !5 {
entry:
  ret i32 42
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "shader.hlsl", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !{!9})
!5 = distinct !DISubprogram(name: "get_value", linkageName: "get_value", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !{})
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
