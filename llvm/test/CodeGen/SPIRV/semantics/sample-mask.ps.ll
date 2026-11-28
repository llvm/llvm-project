; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - | FileCheck %s --implicit-check-not=OpBitcast
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-pixel %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; SV_Coverage is lowered to the SampleMask built-in, which is an array of
; 32-bit integers. The coverage is its first element, which is accessed with an
; access chain. A pointer bitcast is not allowed on Input/Output pointers.

; CHECK-DAG: OpDecorate %[[#IN:]] BuiltIn SampleMask
; CHECK-DAG: OpDecorate %[[#IN]] Flat
; CHECK-DAG: OpDecorate %[[#OUT:]] BuiltIn SampleMask
; CHECK-DAG: %[[#INT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#ZERO:]] = OpConstant %[[#INT]] 0
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#INT]] 1
; CHECK-DAG: %[[#ARR:]] = OpTypeArray %[[#INT]] %[[#ONE]]
; CHECK-DAG: %[[#IN_ARR_PTR:]] = OpTypePointer Input %[[#ARR]]
; CHECK-DAG: %[[#OUT_ARR_PTR:]] = OpTypePointer Output %[[#ARR]]
; CHECK-DAG: %[[#IN_PTR:]] = OpTypePointer Input %[[#INT]]
; CHECK-DAG: %[[#OUT_PTR:]] = OpTypePointer Output %[[#INT]]
; CHECK-DAG: %[[#IN]] = OpVariable %[[#IN_ARR_PTR]] Input
; CHECK-DAG: %[[#OUT]] = OpVariable %[[#OUT_ARR_PTR]] Output

@SV_Coverage = external hidden thread_local addrspace(7) externally_initialized constant [1 x i32], !spirv.Decorations !0
@SV_Coverage.1 = external hidden thread_local addrspace(8) global [1 x i32], !spirv.Decorations !3

define void @main() #0 {
entry:
; CHECK: %[[#IN_ELT:]] = OpInBoundsAccessChain %[[#IN_PTR]] %[[#IN]] %[[#ZERO]]
; CHECK: %[[#VAL:]] = OpLoad %[[#INT]] %[[#IN_ELT]]
  %in.ptr = call ptr addrspace(7) (ptr addrspace(7), ...) @llvm.structured.gep.p7(ptr addrspace(7) elementtype([1 x i32]) @SV_Coverage, i32 0)
  %in = load i32, ptr addrspace(7) %in.ptr, align 4
  %r = and i32 %in, 1
; CHECK: %[[#RES:]] = OpBitwiseAnd %[[#INT]] %[[#VAL]] %[[#ONE]]
; CHECK: %[[#OUT_ELT:]] = OpInBoundsAccessChain %[[#OUT_PTR]] %[[#OUT]] %[[#ZERO]]
; CHECK: OpStore %[[#OUT_ELT]] %[[#RES]]
  %out.ptr = call ptr addrspace(8) (ptr addrspace(8), ...) @llvm.structured.gep.p8(ptr addrspace(8) elementtype([1 x i32]) @SV_Coverage.1, i32 0)
  store i32 %r, ptr addrspace(8) %out.ptr, align 4
  ret void
}

attributes #0 = { "hlsl.shader"="pixel" }

!0 = !{!1, !2}
!1 = !{i32 11, i32 20}
!2 = !{i32 14}
!3 = !{!1}
