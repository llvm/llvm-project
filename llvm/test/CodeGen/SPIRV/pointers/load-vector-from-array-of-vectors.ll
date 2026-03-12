; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[FLOAT:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[VEC2FLOAT:%[0-9]+]] = OpTypeVector [[FLOAT]] 2
; CHECK: OpLoad [[VEC2FLOAT]]
; CHECK: OpCompositeExtract [[FLOAT]]
; CHECK: OpCompositeConstruct [[VEC2FLOAT]]

@M1 = internal addrspace(10) global [4 x <2 x float>] zeroinitializer, align 4
@OUT = internal addrspace(10) global <2 x float> zeroinitializer, align 4

define spir_func void @main() {
entry:
  %0 = load <5 x float>, ptr addrspace(10) @M1, align 4
  %1 = shufflevector <5 x float> %0, <5 x float> poison, <2 x i32> <i32 0, i32 4>
  store <2 x float> %1, ptr addrspace(10) @OUT, align 4
  ret void
}
