; Verify that the NSDI pass does not crash when compiling with debug info for
; a target that does not have SPV_KHR_non_semantic_info enabled (e.g., OpenCL
; targets like spirv64-intel). The pass should silently skip emission.

; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - \
; RUN:   | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-NOT: OpExtension "SPV_KHR_non_semantic_info"

define spir_func void @main() {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cl", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
