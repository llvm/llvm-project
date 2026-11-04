; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[FLOAT:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[VEC4FLOAT:%[0-9]+]] = OpTypeVector [[FLOAT]] 4
; CHECK-DAG: [[UINT_TYPE:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[UINT4:%[0-9]+]] = OpConstant [[UINT_TYPE]] 4
; CHECK-DAG: [[ARRAY4FLOAT:%[0-9]+]] = OpTypeArray [[FLOAT]] [[UINT4]]
; CHECK-DAG: [[PTR_ARRAY4FLOAT:%[0-9]+]] = OpTypePointer Private [[ARRAY4FLOAT]]
; CHECK-DAG: [[G_IN:%[0-9]+]] = OpVariable [[PTR_ARRAY4FLOAT]] Private
; CHECK-DAG: [[G_OUT:%[0-9]+]] = OpVariable [[PTR_ARRAY4FLOAT]] Private
; CHECK-DAG: [[UINT0:%[0-9]+]] = OpConstant [[UINT_TYPE]] 0
; CHECK-DAG: [[UINT1:%[0-9]+]] = OpConstant [[UINT_TYPE]] 1
; CHECK-DAG: [[UINT2:%[0-9]+]] = OpConstant [[UINT_TYPE]] 2
; CHECK-DAG: [[UINT3:%[0-9]+]] = OpConstant [[UINT_TYPE]] 3
; CHECK-DAG: [[PTR_FLOAT:%[0-9]+]] = OpTypePointer Private [[FLOAT]]
; CHECK-DAG: [[UNDEF_VEC:%[0-9]+]] = OpUndef [[VEC4FLOAT]]

@G_in = internal addrspace(10) global [4 x float] zeroinitializer
@G_out = internal addrspace(10) global [4 x float] zeroinitializer

define spir_func void @main() {
entry:
; CHECK:      [[GEP0:%[0-9]+]] = OpAccessChain [[PTR_FLOAT]] [[G_IN]] [[UINT0]]
; CHECK-NEXT: [[LOAD0:%[0-9]+]] = OpLoad [[FLOAT]] [[GEP0]]
; CHECK-NEXT: [[GEP1:%[0-9]+]] = OpAccessChain [[PTR_FLOAT]] [[G_IN]] [[UINT1]]
; CHECK-NEXT: [[LOAD1:%[0-9]+]] = OpLoad [[FLOAT]] [[GEP1]]
; CHECK-NEXT: [[GEP2:%[0-9]+]] = OpAccessChain [[PTR_FLOAT]] [[G_IN]] [[UINT2]]
; CHECK-NEXT: [[LOAD2:%[0-9]+]] = OpLoad [[FLOAT]] [[GEP2]]
; CHECK-NEXT: [[GEP3:%[0-9]+]] = OpAccessChain [[PTR_FLOAT]] [[G_IN]] [[UINT3]]
; CHECK-NEXT: [[LOAD3:%[0-9]+]] = OpLoad [[FLOAT]] [[GEP3]]
; CHECK-NEXT: [[VEC_INSERT0:%[0-9]+]] = OpCompositeInsert [[VEC4FLOAT]] [[LOAD0]] [[UNDEF_VEC]] 0
; CHECK-NEXT: [[VEC_INSERT1:%[0-9]+]] = OpCompositeInsert [[VEC4FLOAT]] [[LOAD1]] [[VEC_INSERT0]] 1
; CHECK-NEXT: [[VEC_INSERT2:%[0-9]+]] = OpCompositeInsert [[VEC4FLOAT]] [[LOAD2]] [[VEC_INSERT1]] 2
; CHECK-NEXT: [[VEC:%[0-9]+]] = OpCompositeInsert [[VEC4FLOAT]] [[LOAD3]] [[VEC_INSERT2]] 3
  %0 = load <4 x float>, ptr addrspace(10) @G_in, align 64

; CHECK-NEXT: [[GEP_OUT0:%[0-9]+]] = OpAccessChain [[PTR_FLOAT]] [[G_OUT]] [[UINT0]]
; CHECK-NEXT: [[VEC_EXTRACT0:%[0-9]+]] = OpCompositeExtract [[FLOAT]] [[VEC]] 0
; CHECK-NEXT: OpStore [[GEP_OUT0]] [[VEC_EXTRACT0]]
; CHECK-NEXT: [[GEP_OUT1:%[0-9]+]] = OpAccessChain [[PTR_FLOAT]] [[G_OUT]] [[UINT1]]
; CHECK-NEXT: [[VEC_EXTRACT1:%[0-9]+]] = OpCompositeExtract [[FLOAT]] [[VEC]] 1
; CHECK-NEXT: OpStore [[GEP_OUT1]] [[VEC_EXTRACT1]]
; CHECK-NEXT: [[GEP_OUT2:%[0-9]+]] = OpAccessChain [[PTR_FLOAT]] [[G_OUT]] [[UINT2]]
; CHECK-NEXT: [[VEC_EXTRACT2:%[0-9]+]] = OpCompositeExtract [[FLOAT]] [[VEC]] 2
; CHECK-NEXT: OpStore [[GEP_OUT2]] [[VEC_EXTRACT2]]
; CHECK-NEXT: [[GEP_OUT3:%[0-9]+]] = OpAccessChain [[PTR_FLOAT]] [[G_OUT]] [[UINT3]]
; CHECK-NEXT: [[VEC_EXTRACT3:%[0-9]+]] = OpCompositeExtract [[FLOAT]] [[VEC]] 3
; CHECK-NEXT: OpStore [[GEP_OUT3]] [[VEC_EXTRACT3]]
  store <4 x float> %0, ptr addrspace(10) @G_out, align 64

; CHECK-NEXT: OpReturn
  ret void
}
