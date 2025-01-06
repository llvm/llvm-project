; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_uniform_group_instructions %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: __spirv_GroupBitwiseAndKHR: the builtin requires the following SPIR-V extension: SPV_KHR_uniform_group_instructions

; CHECK: Capability GroupUniformArithmeticKHR
; CHECK: Extension "SPV_KHR_uniform_group_instructions"
; CHECK-DAG: %[[TyInt:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[TyBool:[0-9]+]] = OpTypeBool
; CHECK-DAG: %[[TyFloat:[0-9]+]] = OpTypeFloat 16
; CHECK-DAG: %[[Scope:[0-9]+]] = OpConstant %[[TyInt]] 2
; CHECK-DAG: %[[ConstInt:[0-9]+]] = OpConstant %[[TyInt]] 0
; CHECK-DAG: %[[ConstFloat:[0-9]+]] = OpConstant %[[TyFloat]] 0
; CHECK-DAG: %[[ConstBool:[0-9]+]] = OpConstantFalse %[[TyBool]]

; CHECK: OpGroupBitwiseAndKHR %[[TyInt]]   %[[Scope]] 0 %[[ConstInt]]
; CHECK: OpGroupBitwiseOrKHR  %[[TyInt]]   %[[Scope]] 0 %[[ConstInt]]
; CHECK: OpGroupBitwiseXorKHR %[[TyInt]]   %[[Scope]] 0 %[[ConstInt]]
; CHECK: OpGroupLogicalAndKHR %[[TyBool]]  %[[Scope]] 0 %[[ConstBool]]
; CHECK: OpGroupLogicalOrKHR  %[[TyBool]]  %[[Scope]] 0 %[[ConstBool]]
; CHECK: OpGroupLogicalXorKHR %[[TyBool]]  %[[Scope]] 0 %[[ConstBool]]
; CHECK: OpGroupIMulKHR       %[[TyInt]]   %[[Scope]] 0 %[[ConstInt]]
; CHECK: OpGroupFMulKHR       %[[TyFloat]] %[[Scope]] 0 %[[ConstFloat]]

; CHECK: OpGroupBitwiseAndKHR %[[TyInt]]   %[[Scope]] 0 %[[ConstInt]]
; CHECK: OpGroupBitwiseOrKHR  %[[TyInt]]   %[[Scope]] 0 %[[ConstInt]]
; CHECK: OpGroupBitwiseXorKHR %[[TyInt]]   %[[Scope]] 0 %[[ConstInt]]
; CHECK: OpGroupLogicalAndKHR %[[TyBool]]  %[[Scope]] 0 %[[ConstBool]]
; CHECK: OpGroupLogicalOrKHR  %[[TyBool]]  %[[Scope]] 0 %[[ConstBool]]
; CHECK: OpGroupLogicalXorKHR %[[TyBool]]  %[[Scope]] 0 %[[ConstBool]]
; CHECK: OpGroupIMulKHR       %[[TyInt]]   %[[Scope]] 0 %[[ConstInt]]
; CHECK: OpGroupFMulKHR       %[[TyFloat]] %[[Scope]] 0 %[[ConstFloat]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define dso_local spir_func void @test1() {
entry:
  %res1 = tail call spir_func i32 @_Z26__spirv_GroupBitwiseAndKHR(i32 2, i32 0, i32 0)
  %res2 = tail call spir_func i32 @_Z25__spirv_GroupBitwiseOrKHR(i32 2, i32 0, i32 0)
  %res3 = tail call spir_func i32 @_Z26__spirv_GroupBitwiseXorKHR(i32 2, i32 0, i32 0)
  %res4 = tail call spir_func i1 @_Z26__spirv_GroupLogicalAndKHR(i32 2, i32 0, i1 false)
  %res5 = tail call spir_func i1 @_Z25__spirv_GroupLogicalOrKHR(i32 2, i32 0, i1 false)
  %res6 = tail call spir_func i1 @_Z26__spirv_GroupLogicalXorKHR(i32 2, i32 0, i1 false)
  %res7 = tail call spir_func i32 @_Z20__spirv_GroupIMulKHR(i32 2, i32 0, i32 0)
  %res8 = tail call spir_func half @_Z20__spirv_GroupFMulKHR(i32 2, i32 0, half 0xH0000)
  ret void
}

define dso_local spir_func void @test2() {
entry:
  %res1 = tail call spir_func i32  @_Z21work_group_reduce_andi(i32 0)
  %res2 = tail call spir_func i32  @_Z20work_group_reduce_ori(i32 0)
  %res3 = tail call spir_func i32  @_Z21work_group_reduce_xori(i32 0)
  %res4 = tail call spir_func i32  @_Z29work_group_reduce_logical_andi(i32 0)
  %res5 = tail call spir_func i32  @_Z28work_group_reduce_logical_ori(i32 0)
  %res6 = tail call spir_func i32  @_Z29work_group_reduce_logical_xori(i32 0)
  %res7 = tail call spir_func i32  @_Z21work_group_reduce_muli(i32 0)
  %res8 = tail call spir_func half @_Z21work_group_reduce_mulDh(half 0xH0000)
  ret void
}

declare dso_local spir_func i32  @_Z26__spirv_GroupBitwiseAndKHR(i32, i32, i32)
declare dso_local spir_func i32  @_Z25__spirv_GroupBitwiseOrKHR(i32, i32, i32)
declare dso_local spir_func i32  @_Z26__spirv_GroupBitwiseXorKHR(i32, i32, i32)
declare dso_local spir_func i1   @_Z26__spirv_GroupLogicalAndKHR(i32, i32, i1)
declare dso_local spir_func i1   @_Z25__spirv_GroupLogicalOrKHR(i32, i32, i1)
declare dso_local spir_func i1   @_Z26__spirv_GroupLogicalXorKHR(i32, i32, i1)
declare dso_local spir_func i32  @_Z20__spirv_GroupIMulKHR(i32, i32, i32)
declare dso_local spir_func half @_Z20__spirv_GroupFMulKHR(i32, i32, half)

declare dso_local spir_func i32  @_Z21work_group_reduce_andi(i32)
declare dso_local spir_func i32  @_Z20work_group_reduce_ori(i32)
declare dso_local spir_func i32  @_Z21work_group_reduce_xori(i32)
declare dso_local spir_func i32  @_Z29work_group_reduce_logical_andi(i32)
declare dso_local spir_func i32  @_Z28work_group_reduce_logical_ori(i32)
declare dso_local spir_func i32  @_Z29work_group_reduce_logical_xori(i32)
declare dso_local spir_func i32  @_Z21work_group_reduce_muli(i32)
declare dso_local spir_func half @_Z21work_group_reduce_mulDh(half)
