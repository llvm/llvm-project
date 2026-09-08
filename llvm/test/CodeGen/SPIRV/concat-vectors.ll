; RUN: llc -O0 -global-isel -verify-machineinstrs -mtriple=spirv64 --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64 --spirv-ext=+SPV_EXT_long_vector < %s -o - -filetype=obj | spirv-val %}
; RUN: not llc -O0 -global-isel -verify-machineinstrs -mtriple=spirv64 %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; Note we have to use a non-power of 2 vector length (7 here) that cannot be legalized.
; Usually it would be legalized but here it can't because it's part of the ABI.

; CHECK: %[[#I8:]] = OpTypeInt 8 0
; CHECK: %[[#V4:]] = OpTypeVector %[[#I8]] 4
; CHECK: %[[#I32:]] = OpTypeInt 32 0
; CHECK: %[[#SEVEN:]] = OpConstant %[[#I32]] 7
; CHECK: %[[#V7:]] = OpTypeVectorIdEXT %[[#I8]] %[[#SEVEN]]
; CHECK: %[[#UNDEF:]] = OpUndef %[[#V4]]
; CHECK: OpFunction
; CHECK: %[[#A:]] = OpFunctionParameter %[[#V4]]
; CHECK: OpVectorShuffle %[[#V7]] %[[#A]] %[[#UNDEF]] 0 1 2 3

; CHECK-ERROR: LLVM ERROR: OpTypeVector with 7 components requires the following SPIR-V extension: SPV_EXT_long_vector

define spir_func <7 x i8> @extend_vec4_to_vec7(<4 x i8> %x) {

entry:
  %r = shufflevector <4 x i8> %x, <4 x i8> poison, <7 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison>
  ret <7 x i8> %r
}
