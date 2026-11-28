; RUN: llc -O0 -mtriple=spirv64v1.6-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CORE
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.6-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefixes=CHECK,EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - -filetype=obj | spirv-val %}
; Default OpenCL version (1.4) without the extension falls back to OpNop.
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=NOEXT

; CORE-NOT:  OpExtension "SPV_KHR_non_semantic_info"
; EXT:       OpExtension "SPV_KHR_non_semantic_info"
; CHECK:     %[[#Set:]] = OpExtInstImport "NonSemantic.DebugBreak"
; CHECK:     OpExtInst %[[#]] %[[#Set]] DebugBreak
; CHECK:     OpReturn

; On the AMD triple llvm.debugtrap is lowered to a call to an imported function
; by SPIRVPrepareFunctions, before instruction selection, so it never becomes DebugBreak.
; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s --check-prefix=AMD
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}
; AMD-NOT: OpExtInstImport "NonSemantic.DebugBreak"
; AMD:     OpDecorate %[[#F:]] LinkageAttributes "spirv.llvm_debugtrap" Import
; AMD:     OpFunctionCall %[[#]] %[[#F]]
; AMD-NOT: DebugBreak
; AMD-NOT: OpNop
; AMD:     OpReturn

; NOEXT-NOT: OpExtInstImport "NonSemantic.DebugBreak"
; NOEXT:     OpNop
; NOEXT:     OpReturn

define void @test_kernel() {
  call void @llvm.debugtrap()
  ret void
}

declare void @llvm.debugtrap()
