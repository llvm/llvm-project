; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#I32]]
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#I64]] 1
; CHECK-DAG: %[[#TWO:]] = OpConstant %[[#I64]] 2
; CHECK-DAG: %[[#FIVE:]] = OpConstant %[[#I64]] 5
; CHECK-DAG: %[[#VPTR2:]] = OpTypeVector %[[#PTR]] 2
; CHECK-DAG: %[[#VI64_2:]] = OpTypeVector %[[#I64]] 2
; CHECK-DAG: %[[#UNDEF2:]] = OpUndef %[[#VPTR2]]
; CHECK-DAG: %[[#NULL2:]] = OpConstantNull %[[#VI64_2]]
; CHECK-DAG: %[[#VPTR4:]] = OpTypeVector %[[#PTR]] 4
; CHECK-DAG: %[[#VI64_4:]] = OpTypeVector %[[#I64]] 4
; CHECK-DAG: %[[#UNDEF4:]] = OpUndef %[[#VPTR4]]
; CHECK-DAG: %[[#NULL4:]] = OpConstantNull %[[#VI64_4]]

; The <1 x ptr> GEP collapses to a single scalar OpPtrAccessChain; no
; vector-of-pointers value is materialized.
; CHECK:      OpFunction
; CHECK-NEXT: %[[#P1:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: %[[#OUT1:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#GEP1:]] = OpPtrAccessChain %[[#PTR]] %[[#P1]] %[[#FIVE]]
; CHECK-NEXT: %[[#VAL1:]] = OpLoad %[[#I32]] %[[#GEP1]]
; CHECK-NEXT: OpStore %[[#OUT1]] %[[#VAL1]]
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
define spir_kernel void @test_vector_gep_v1(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <1 x i64> <i64 5>
  %elem = extractelement <1 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK:      OpFunction
; CHECK-NEXT: %[[#P2:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: %[[#OUT2:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#IDX2_0:]] = OpCompositeExtract %[[#I64]] %[[#NULL2]] 0
; CHECK-NEXT: %[[#GEP2_0:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX2_0]]
; CHECK-NEXT: %[[#INS2_0:]] = OpCompositeInsert %[[#VPTR2]] %[[#GEP2_0]] %[[#UNDEF2]] 0
; CHECK-NEXT: %[[#IDX2_1:]] = OpCompositeExtract %[[#I64]] %[[#NULL2]] 1
; CHECK-NEXT: %[[#GEP2_1:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX2_1]]
; CHECK-NEXT: %[[#INS2_1:]] = OpCompositeInsert %[[#VPTR2]] %[[#GEP2_1]] %[[#INS2_0]] 1
; CHECK-NEXT: %[[#ELT2:]] = OpCompositeExtract %[[#PTR]] %[[#INS2_1]] 0
; CHECK-NEXT: %[[#VAL2:]] = OpLoad %[[#I32]] %[[#ELT2]]
; CHECK-NEXT: OpStore %[[#OUT2]] %[[#VAL2]]
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
define spir_kernel void @test_vector_gep_v2(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <2 x i64> zeroinitializer
  %elem = extractelement <2 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK:      OpFunction
; CHECK-NEXT: %[[#P4:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: %[[#OUT4:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#IDX4_0:]] = OpCompositeExtract %[[#I64]] %[[#NULL4]] 0
; CHECK-NEXT: %[[#GEP4_0:]] = OpPtrAccessChain %[[#PTR]] %[[#P4]] %[[#IDX4_0]]
; CHECK-NEXT: %[[#INS4_0:]] = OpCompositeInsert %[[#VPTR4]] %[[#GEP4_0]] %[[#UNDEF4]] 0
; CHECK-NEXT: %[[#IDX4_1:]] = OpCompositeExtract %[[#I64]] %[[#NULL4]] 1
; CHECK-NEXT: %[[#GEP4_1:]] = OpPtrAccessChain %[[#PTR]] %[[#P4]] %[[#IDX4_1]]
; CHECK-NEXT: %[[#INS4_1:]] = OpCompositeInsert %[[#VPTR4]] %[[#GEP4_1]] %[[#INS4_0]] 1
; CHECK-NEXT: %[[#IDX4_2:]] = OpCompositeExtract %[[#I64]] %[[#NULL4]] 2
; CHECK-NEXT: %[[#GEP4_2:]] = OpPtrAccessChain %[[#PTR]] %[[#P4]] %[[#IDX4_2]]
; CHECK-NEXT: %[[#INS4_2:]] = OpCompositeInsert %[[#VPTR4]] %[[#GEP4_2]] %[[#INS4_1]] 2
; CHECK-NEXT: %[[#IDX4_3:]] = OpCompositeExtract %[[#I64]] %[[#NULL4]] 3
; CHECK-NEXT: %[[#GEP4_3:]] = OpPtrAccessChain %[[#PTR]] %[[#P4]] %[[#IDX4_3]]
; CHECK-NEXT: %[[#INS4_3:]] = OpCompositeInsert %[[#VPTR4]] %[[#GEP4_3]] %[[#INS4_2]] 3
; CHECK-NEXT: %[[#ELT4:]] = OpCompositeExtract %[[#PTR]] %[[#INS4_3]] 0
; CHECK-NEXT: %[[#VAL4:]] = OpLoad %[[#I32]] %[[#ELT4]]
; CHECK-NEXT: OpStore %[[#OUT4]] %[[#VAL4]]
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
define spir_kernel void @test_vector_gep_v4(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <4 x i64> zeroinitializer
  %elem = extractelement <4 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK:      OpFunction
; CHECK-NEXT: %[[#PV:]] = OpFunctionParameter %[[#VPTR2]]
; CHECK-NEXT: %[[#OUTV:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#EXPV_0:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 0
; CHECK-NEXT: %[[#GEPV_0:]] = OpPtrAccessChain %[[#PTR]] %[[#EXPV_0]] %[[#ONE]]
; CHECK-NEXT: %[[#INSV_0:]] = OpCompositeInsert %[[#VPTR2]] %[[#GEPV_0]] %[[#UNDEF2]] 0
; CHECK-NEXT: %[[#EXPV_1:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 1
; CHECK-NEXT: %[[#GEPV_1:]] = OpPtrAccessChain %[[#PTR]] %[[#EXPV_1]] %[[#TWO]]
; CHECK-NEXT: %[[#INSV_1:]] = OpCompositeInsert %[[#VPTR2]] %[[#GEPV_1]] %[[#INSV_0]] 1
; CHECK-NEXT: %[[#ELTV:]] = OpCompositeExtract %[[#PTR]] %[[#INSV_1]] 0
; CHECK-NEXT: %[[#VALV:]] = OpLoad %[[#I32]] %[[#ELTV]]
; CHECK-NEXT: OpStore %[[#OUTV]] %[[#VALV]]
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
define spir_kernel void @test_vector_gep_vec_ptr(<2 x ptr addrspace(1)> %ptrs, ptr addrspace(1) %out) {
  %gep = getelementptr i32, <2 x ptr addrspace(1)> %ptrs, <2 x i64> <i64 1, i64 2>
  %elem = extractelement <2 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}
