; Verify that the NSDI pass activates automatically when the module contains
; debug info (llvm.dbg.cu) without requiring --spv-emit-nonsemantic-debug-info.
; The language constant in DebugCompilationUnit must be 5 (HLSL).

; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info \
; RUN:   -O0 -mtriple=spirv64-unknown-unknown %s -o - \
; RUN:   | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info \
; RUN:   -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj \
; RUN:   | spirv-val %}

; CHECK: OpExtension "SPV_KHR_non_semantic_info"
; CHECK: [[ext_inst:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[type_i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[lang_hlsl:%[0-9]+]] = OpConstant [[type_i32]] 5
; CHECK: OpExtInst [[type_void]] [[ext_inst]] DebugCompilationUnit
; CHECK-SAME: [[lang_hlsl]]

define spir_func void @main() {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "shader.hlsl", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
