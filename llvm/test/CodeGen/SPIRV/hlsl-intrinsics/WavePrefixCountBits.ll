; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; Test lowering to spir-v backend

define noundef i32 @wave_prefix_count_bits(i1 noundef %expr) {
entry:
  ; CHECK: %[[UINT:.*]] = OpTypeInt 32 0
  ; CHECK: %[[UINT4:.*]] = OpTypeVector %[[UINT]] 4
  ; CHECK: %[[UINT3:.*]] = OpConstant %[[UINT]] 3
  ; CHECK: %[[INPUTREG:.*]] = OpFunctionParameter
  ; CHECK: %[[BALLOTRESULT:.*]] = OpGroupNonUniformBallot %[[UINT4]] %[[UINT3]] %[[INPUTREG]]
  ; CHECK: %[[RET:.*]] = OpGroupNonUniformBallotBitCount %[[UINT]] %[[UINT3]] ExclusiveScan %[[BALLOTRESULT]]
  %ret = call i32 @llvm.spv.subgroup.prefix.bit.count(i1 %expr)
  ; CHECK: OpReturnValue %[[RET]]
  ret i32 %ret
}
