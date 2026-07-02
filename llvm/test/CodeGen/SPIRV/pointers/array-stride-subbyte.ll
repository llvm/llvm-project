; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-vulkan1.3-library %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-vulkan1.3-library %s -o - -filetype=obj | spirv-val %}

; A sub-byte element (i1) must not truncate its ArrayStride to 0.

@.str = private unnamed_addr constant [2 x i8] c"B\00", align 1

; CHECK: OpDecorate [[array:%[0-9]+]] ArrayStride 1
; CHECK: [[bool:%[0-9]+]] = OpTypeBool
; CHECK: [[array]] = OpTypeRuntimeArray [[bool]]

define external void @main() {
  %handle = tail call target("spirv.VulkanBuffer", [0 x i1], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  ret void
}
