; RUN: llc -verify-machineinstrs -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; spirv-val does not yet handle long vectors of pointers even though the rules are the same as for OpTypeVector
; TODO: %if spirv-tools %{ llc -O0 --spirv-ext=+SPV_EXT_long_vector -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: %[[#I32:]] = OpTypeInt 32
; CHECK: %[[#V17I32:]] = OpTypeVectorIdEXT %[[#I32]] 17
; CHECK: OpTypeFunction
; CHECK: %[[#X17I32:]] = OpFunctionParameter %[[#V17I32]]
; CHECK: %[[#]] = OpBitCount %[[#V17I32]] %[[#X17I32]]

@g5 = addrspace(1) global <17 x i32> undef, align 4

define dso_local spir_kernel void @test(<17 x i32> %x17i32) {
entry:
  %4 = tail call <17 x i32> @llvm.ctpop.v17i32(<17 x i32> %x17i32)
  store <17 x i32> %4, ptr addrspace(1) @g5, align 4

  ret void
}

declare <17 x i32> @llvm.ctpop.v17i32(<17 x i32>)
