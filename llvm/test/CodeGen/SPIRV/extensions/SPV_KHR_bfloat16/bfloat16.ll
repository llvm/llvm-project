; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bfloat16 %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: OpTypeFloat type with bfloat requires the following SPIR-V extension: SPV_KHR_bfloat16

; CHECK-DAG: OpCapability BFloat16TypeKHR
; CHECK-DAG: OpExtension "SPV_KHR_bfloat16"
; CHECK: %[[#BFLOAT:]] = OpTypeFloat 16 0
; CHECK: %[[#]] = OpTypeVector %[[#BFLOAT]] 2

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test() {
entry:
  %addr1 = alloca bfloat
  %addr2 = alloca <2 x bfloat>
  %data1 = load bfloat, ptr %addr1
  %data2 = load <2 x bfloat>, ptr %addr2
  ret void
}
