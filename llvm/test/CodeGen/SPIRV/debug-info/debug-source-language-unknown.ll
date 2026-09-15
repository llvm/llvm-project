; Verify that an unrecognized DWARF source language produces source language
; code 0 (Unknown) in DebugCompilationUnit. SPIRVNonSemanticDebugHandler maps
; any language not explicitly listed in toNSDISrcLang() to 0 rather than
; failing. DW_LANG_C_plus_plus_14 is used here as a representative unmapped
; language.

; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: [[ext_inst:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[type_void:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[type_i32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[lang_unknown:%[0-9]+]] = OpConstant [[type_i32]] 0{{$}}
; CHECK: OpExtInst [[type_void]] [[ext_inst]] DebugCompilationUnit
; CHECK-SAME: [[lang_unknown]]

define spir_func void @main() {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
