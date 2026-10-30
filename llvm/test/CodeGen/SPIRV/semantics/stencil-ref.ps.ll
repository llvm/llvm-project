; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-pixel -spirv-ext=+SPV_EXT_shader_stencil_export %s -o - | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel -spirv-ext=+SPV_EXT_shader_stencil_export %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; SV_StencilRef is lowered to a store to the FragStencilRefEXT built-in, which
; requires the StencilExportEXT capability from SPV_EXT_shader_stencil_export.
; ERROR: LLVM ERROR: FragStencilRefEXT requires the following SPIR-V extension: SPV_EXT_shader_stencil_export

; CHECK-DAG: OpCapability StencilExportEXT
; CHECK-DAG: OpExtension "SPV_EXT_shader_stencil_export"
; CHECK-DAG: OpDecorate %[[#STENCIL:]] BuiltIn FragStencilRefEXT
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer Output %[[#INT]]
; CHECK-DAG: %[[#STENCIL]] = OpVariable %[[#PTR]] Output

@SV_StencilRef = external hidden thread_local addrspace(8) global i32, !spirv.Decorations !0

define void @main() #0 {
entry:
; CHECK: OpStore %[[#STENCIL]] %[[#]]
  store i32 3, ptr addrspace(8) @SV_StencilRef, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{!1}
!1 = !{i32 11, i32 5014}
