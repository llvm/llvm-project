; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Kernel
; CHECK: OpCapability Addresses
; CHECK: OpCapability GenericPointer
; CHECK: OpCapability Int64
; CHECK: OpCapability Int8
; CHECK: OpCapability Linkage

; CHECK: OpExtInstImport "OpenCL.std"
; CHECK: OpMemoryModel Physical64 OpenCL
; CHECK: OpEntryPoint Kernel %[[#]] "foo" %[[#]]
; CHECK: OpSource OpenCL_C 200000

; CHECK-DAG: OpName %[[#]]
; CHECK-DAG: OpDecorate %[[#]]


; CHECK: %[[#I8:]] = OpTypeInt 8 0
; CHECK: %[[#PTR_CW_I8:]] = OpTypePointer CrossWorkgroup %[[#I8]]
; CHECK: %[[#I32:]] = OpTypeInt 32 0
; CHECK: %[[#VEC4:]] = OpTypeVector %[[#I32]] 4
; CHECK: %[[#VOID:]] = OpTypeVoid
; CHECK: %[[#FUNC_TYPE0:]] = OpTypeFunction %[[#VOID]] %[[#PTR_CW_I8]] %[[#VEC4]]
; CHECK: %[[#FUNC_TYPE1:]] = OpTypeFunction %[[#VOID]] %[[#PTR_CW_I8]]
; CHECK: %[[#VEC3:]] = OpTypeVector %[[#I32]] 3
; CHECK: %[[#FUNC_TYPE2:]] = OpTypeFunction %[[#VOID]] %[[#PTR_CW_I8]] %[[#VEC3]]
; CHECK: %[[#PTR_GEN_I8:]] = OpTypePointer Generic %[[#I8]]
; CHECK: %[[#STRUCT_B:]] = OpTypeStruct %[[#I32]] %[[#PTR_GEN_I8]]
; CHECK: %[[#STRUCT_C:]] = OpTypeStruct %[[#I32]] %[[#STRUCT_B]]
; CHECK: %[[#STRUCT_A:]] = OpTypeStruct %[[#I32]] %[[#STRUCT_C]]
; CHECK: %[[#F32:]] = OpTypeFloat 32
; CHECK: %[[#CONST_2:]] = OpConstant %[[#I32]] 2
; CHECK: %[[#ARRAY_F:]] = OpTypeArray %[[#F32]] %[[#CONST_2]]
; CHECK: %[[#ARRAY_I:]] = OpTypeArray %[[#I32]] %[[#CONST_2]]
; CHECK: %[[#PTR_CW_STRUCT_A:]] = OpTypePointer CrossWorkgroup %[[#STRUCT_A]]
; CHECK: %[[#PTR_UC_VEC4:]] = OpTypePointer UniformConstant %[[#VEC4]]
; CHECK: %[[#PTR_UC_ARRAY_F:]] = OpTypePointer UniformConstant %[[#ARRAY_F]]
; CHECK: %[[#PTR_CW_PTR_CW_I8:]] = OpTypePointer CrossWorkgroup %[[#PTR_CW_I8]]
; CHECK: %[[#I64:]] = OpTypeInt 64 0
; CHECK: %[[#PTR_CW_ARRAY_I:]] = OpTypePointer CrossWorkgroup %[[#ARRAY_I]]

; CHECK: %[[#NULL_I32:]] = OpConstantNull %[[#I32]]
; CHECK: %[[#CONST_I64_4:]] = OpConstant %[[#I64]] 4
; CHECK: %[[#CONST_I32_1:]] = OpConstant %[[#I32]] 1
; CHECK: %[[#COMP_I32:]] = OpConstantComposite %[[#ARRAY_I]] %[[#CONST_I32_1]] %[[#CONST_2]]

; CHECK: %[[#VAR_V:]] = OpVariable %[[#PTR_CW_ARRAY_I]] CrossWorkgroup %[[#COMP_I32]]
; CHECK: %[[#SPECCONSTOP:]] = OpSpecConstantOp %[[#PTR_CW_I8]] InBoundsPtrAccessChain %[[#VAR_V]] %[[#NULL_I32]] %[[#CONST_I64_4]]
; CHECK: %[[#VAR_S:]] = OpVariable %[[#PTR_CW_PTR_CW_I8]] CrossWorkgroup %[[#SPECCONSTOP]]
; CHECK: %[[#NULL_ARRAY_F:]] = OpConstantNull %[[#ARRAY_F]]
; CHECK: %[[#VAR_F:]] = OpVariable %[[#PTR_UC_ARRAY_F]] UniformConstant %[[#NULL_ARRAY_F]]
; CHECK: %[[#NULL_STRUCT_A:]] = OpConstantNull %[[#STRUCT_A]]
; CHECK: %[[#VAR_A:]] = OpVariable %[[#PTR_CW_STRUCT_A]] CrossWorkgroup %[[#NULL_STRUCT_A]]

; CHECK: %[[#FN_BAR1:]] = OpFunction %[[#VOID]] None %[[#FUNC_TYPE1]]
; CHECK: %[[#P_BAR1:]] = OpFunctionParameter %[[#PTR_CW_I8]]
; CHECK: OpFunctionEnd

@v = addrspace(1) global [2 x i32] [i32 1, i32 2], align 4
@s = addrspace(1) global ptr addrspace(1) getelementptr inbounds ([2 x i32], ptr addrspace(1) @v, i32 0, i32 1), align 4

%struct.A = type { i32, %struct.C }
%struct.C = type { i32, %struct.B }
%struct.B = type { i32, ptr addrspace(4) }

@f = addrspace(2) constant [2 x float] zeroinitializer, align 4
@b = external addrspace(2) constant <4 x i32>
@a = common addrspace(1) global %struct.A zeroinitializer, align 4

define spir_kernel void @foo(ptr addrspace(1) %a, <4 x i32> %vec_in) {
entry:
  call spir_func void @bar1(ptr addrspace(1) %a)
  %extractVec = shufflevector <4 x i32> %vec_in, <4 x i32> %vec_in, <3 x i32> <i32 0, i32 1, i32 2>
  call spir_func void @bar2(ptr addrspace(1) %a, <3 x i32> %extractVec)
  ret void
}

declare spir_func void @bar1(ptr addrspace(1))
declare spir_func void @bar2(ptr addrspace(1), <3 x i32>)

!opencl.ocl.version = !{!7}
!7 = !{i32 2, i32 0}
