; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-amd-amdhsa %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-amd-amdhsa %s -o - -filetype=obj | spirv-val %}

target triple = "spirv64-amd-amdhsa"

@buf = external addrspace(8) global i8
@flat = external addrspace(1) global i8

; CHECK: OpName %[[#NULL:]] "null"
; CHECK: OpName %[[#NULL_VECTOR:]] "null_vector"
; CHECK: OpName %[[#POISON:]] "poison"
; CHECK: OpName %[[#POISON_VEC:]] "poison_vec"
; CHECK: OpName %[[#CAST_GLOBAL:]] "cast_global"
; CHECK: OpName %[[#BUF:]] "buf"
; CHECK: OpName %[[#OPAQUE_PTR_CAST_P8_P7:]] "spirv.llvm_spv_opaque_ptr_cast_p7_p8"
; CHECK: OpName %[[#CAST_NULL:]] "cast_null"
; CHECK: OpName %[[#CAST_VEC:]] "cast_vec"
; CHECK: OpName %[[#GEP:]] "gep"
; CHECK: OpName %[[#GEP_VECTOR:]] "gep_vector"
; CHECK: OpName %[[#GEP_OF_P7:]] "gep_of_p7"
; CHECK: OpName %[[#FLAT:]] "flat"
; CHECK: OpName %[[#GEP_OF_P7_VECTOR:]] "gep_of_p7_vector"
; CHECK: OpName %[[#GEP_OF_P7_STRUCT:]] "gep_of_p7_struct"
; CHECK: OpName %[[#GEP_P7_FROM_P7:]] "gep_p7_from_p7"
; CHECK: OpName %[[#PTRTOINT:]] "ptrtoint"
; CHECK: OpName %[[#PTRTOINT_LONG:]] "ptrtoint_long"
; CHECK: OpName %[[#PTRTOINT_SHORT:]] "ptrtoint_short"
; CHECK: OpName %[[#PTRTOINT_VERY_SHORT:]] "ptrtoint_very_short"
; CHECK: OpName %[[#PTRTOINT_VEC:]] "ptrtoint_vec"
; CHECK: OpName %[[#INTTOPTR:]] "inttoptr"
; CHECK: OpName %[[#INTTOPTR_VEC:]] "inttoptr_vec"
; CHECK: OpName %[[#FANCY_ZERO:]] "fancy_zero"
; CHECK: OpName %[[#LOAD_NULL:]] "load_null"
; CHECK: OpName %[[#STORE_NULL:]] "store_null"
; CHECK: OpName %[[#LOAD_POISON:]] "load_poison"
; CHECK: OpName %[[#STORE_POISON:]] "store_poison"

; CHECK: %[[#INT8_TY:]] = OpTypeInt 8
; CHECK: %[[#I8PTR_ADDRSPACE_7:]] = OpTypePointer DeviceOnlyINTEL %[[#INT8_TY]]
; CHECK: %[[#INT32_TY:]] = OpTypeInt 32
; CHECK: %[[#I32_PTR_GLOBAL:]] = OpTypePointer CrossWorkgroup %[[#INT32_TY]]
; CHECK: %[[#NULLPTR_I8PTR_ADDRSPACE_7:]] = OpConstantNull %[[#I8PTR_ADDRSPACE_7]]
	; %8 = OpConstantNull %5
	; %9 = OpVariable %6 CrossWorkgroup %8
; CHECK: %[[#VEC2_I8PTR_ADDRSPACE_7:]] = OpTypeVector %[[#I8PTR_ADDRSPACE_7]] 2
; CHECK: %[[#INT160_TY:]] = OpTypeInt 160
; CHECK: %[[#ZERO_INT160:]] = OpConstantNull %[[#INT160_TY]]
; CHECK: %[[#NULLVEC2_I8PTR_ADDRSPACE_7:]] = OpConstantComposite %[[#VEC2_I8PTR_ADDRSPACE_7]] %[[#ZERO_INT160]] %[[#ZERO_INT160]]
; CHECK: %[[#UNDEF_I8PTR_ADDRSPACE_7:]] = OpUndef %[[#I8PTR_ADDRSPACE_7]]
; CHECK: %[[#UNDEF_VEC2_I8PTR_ADDRSPACE_7:]] = OpUndef %[[#VEC2_I8PTR_ADDRSPACE_7]]
; CHECK: %[[#I8PTR_ADDRSPACE_8:]] = OpTypePointer HostOnlyINTEL %[[#INT8_TY]]
	; %18 = OpTypeFunction %3 %17
; CHECK: %[[#BUF:]] = OpVariable %[[#I8PTR_ADDRSPACE_8]] HostOnlyINTEL
; CHECK: %[[#NULLPTR_I8PTR_ADDRSPACE_8:]] = OpConstantNull %[[#I8PTR_ADDRSPACE_8]]
; CHECK: %[[#I32PTR_ADDRSPACE_7:]] = OpTypePointer DeviceOnlyINTEL %[[#INT32_TY]]
	; %24 = OpTypeFunction %23
	; %25 = OpConstant %5 4
; CHECK: %[[#ARR_TY:]] = OpTypeArray %[[#INT32_TY]]
; CHECK: %[[#ARRPTR_ADDRSPACE_7:]] = OpTypePointer DeviceOnlyINTEL %[[#ARR_TY]]
; CHECK: %[[#INT64_TY:]] = OpTypeInt 64
	; %29 = OpConstant %5 1
	; %30 = OpConstant %28 2
; CHECK: %[[#VEC2_I32:]] = OpTypeVector %[[#INT32_TY]] 2
	; %32 = OpConstant %5 3
; CHECK: %[[#CONSTEXPR_CAST_PTR_AS8_TO_INT:]] = OpSpecConstantOp %[[#INT64_TY]] ConvertPtrToU %[[#BUF]]
; CHECK: %[[#CONSTEXPR_CAST_INT_AS8_PTR_TO_PTR_AS7:]] = OpSpecConstantOp %[[#I8PTR_ADDRSPACE_7]] ConvertUToPtr %[[#CONSTEXPR_CAST_PTR_AS8_TO_INT]]
	; %35 = OpConstantComposite %31 %32 %29
; CHECK: %[[#TEMP_VEC_WITH_CONSTEXPR_AS_CAST:]] = OpSpecConstantComposite %[[#VEC2_I32]] %[[#CONSTEXPR_CAST_INT_AS8_PTR_TO_PTR_AS7]]
; CHECK: %[[#I8PTR_ADDRSPACE_7PTR_GLOBAL:]] = OpTypePointer CrossWorkgroup %[[#I8PTR_ADDRSPACE_7]]
; CHECK: %[[#I8PTR_GLOBAL:]] = OpTypePointer CrossWorkgroup %[[#INT8_TY]]
; CHECK: %[[#FLAT:]] = OpVariable %[[#I8PTR_GLOBAL]] CrossWorkgroup
; CHECK: %[[#VEC2_I8PTR_ADDRSPACE_7PTR_GLOBAL:]] = OpTypePointer CrossWorkgroup %[[#VEC2_I8PTR_ADDRSPACE_7]]
; CHECK: %[[#STRUCT_TY:]] = OpTypeStruct %[[#I8PTR_ADDRSPACE_7]] %[[#INT32_TY]]
; CHECK: %[[#STRUCTPTR_GLOBAL:]] = OpTypePointer CrossWorkgroup %[[#STRUCT_TY]]
; CHECK: %[[#PTRI8PTR_ADDRSPACE_7_ADDRSPACE_7:]] = OpTypePointer DeviceOnlyINTEL %[[#I8PTR_ADDRSPACE_7]]
; CHECK: %[[#INT256_TY:]] = OpTypeInt 256
; CHECK: %[[#VEC2_I160_TY:]] = OpTypeVector %[[#INT160_TY]] 2
; CHECK: %[[#NULL_VEC2_I160:]] = OpConstantNull %[[#VEC2_I160_TY]]
; CHECK: %[[#CONSTEXPR_CAST_INT_1_TO_PTR_AS7:]] = OpSpecConstantOp %3 ConvertUToPtr
; CHECK: %[[#CONSTEXPR_CAST_INT_2_TO_PTR_AS7:]] = OpSpecConstantOp %3 ConvertUToPtr

; CHECK: %[[#NULL]] = OpFunction %[[#I8PTR_ADDRSPACE_7]]
;	CHECK: OpReturnValue %[[#NULLPTR_I8PTR_ADDRSPACE_7]]
define spir_func ptr addrspace(7) @null() {
  ret ptr addrspace(7) null
}

; CHECK: %[[#NULL_VECTOR]] = OpFunction %[[#VEC2_I8PTR_ADDRSPACE_7]]
;	CHECK: OpReturnValue %[[#NULLVEC2_I8PTR_ADDRSPACE_7]]
define spir_func <2 x ptr addrspace(7)> @null_vector() {
  ret <2 x ptr addrspace(7)> zeroinitializer
}

; CHECK: %[[#POISON]] = OpFunction %[[#I8PTR_ADDRSPACE_7]]
;	CHECK: OpReturnValue %[[#UNDEF_I8PTR_ADDRSPACE_7]]
define spir_func ptr addrspace(7) @poison() {
  ret ptr addrspace(7) poison
}

; CHECK: %[[#POISON_VEC]] = OpFunction %[[#VEC2_I8PTR_ADDRSPACE_7]]
;	CHECK: OpReturnValue %[[#UNDEF_VEC2_I8PTR_ADDRSPACE_7]]
define spir_func <2 x ptr addrspace(7)> @poison_vec() {
  ret <2 x ptr addrspace(7)> poison
}

; CHECK: %[[#CAST_GLOBAL]] = OpFunction %[[#I8PTR_ADDRSPACE_7]]
; CHECK: %[[#CAST_FROM_AS8_TO_AS7:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST_P8_P7]] %[[#BUF]]
;	CHECK: OpReturnValue %[[#CAST_FROM_AS8_TO_AS7]]
define spir_func ptr addrspace(7) @cast_global() {
  ret ptr addrspace(7) addrspacecast (ptr addrspace(8) @buf to ptr addrspace(7))
}

; CHECK: %[[#CAST_NULL]] = OpFunction %[[#I8PTR_ADDRSPACE_7]]
; CHECK: %[[#CAST_NULL_FROM_AS8_TO_AS7:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST_P8_P7]] %[[#NULLPTR_I8PTR_ADDRSPACE_8]]
;	CHECK: OpReturnValue %[[#CAST_NULL_FROM_AS8_TO_AS7]]
define spir_func ptr addrspace(7) @cast_null() {
  ret ptr addrspace(7) addrspacecast (ptr addrspace(8) null to ptr addrspace(7))
}

; CHECK: %[[#CAST_VEC]] = OpFunction %[[#VEC2_I8PTR_ADDRSPACE_7]]
; CHECK: %[[#AS8_TO_AS7:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST_P8_P7]] %[[#BUF]]
; CHECK: %[[#NULL_AS8_TO_AS7:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST_P8_P7]] %[[#NULLPTR_I8PTR_ADDRSPACE_8]]
;	CHECK: %[[#V0:]] = OpCompositeInsert %[[#VEC2_I8PTR_ADDRSPACE_7]] %[[#AS8_TO_AS7]] %[[#UNDEF_VEC2_I8PTR_ADDRSPACE_7]] 0
;	CHECK: %[[#V:]] = OpCompositeInsert %[[#VEC2_I8PTR_ADDRSPACE_7]] %[[#NULL_AS8_TO_AS7]] %[[#V0]] 1
;	CHECK: OpReturnValue %[[#V]]
define spir_func <2 x ptr addrspace(7)> @cast_vec() {
  ret <2 x ptr addrspace(7)> addrspacecast (
  <2 x ptr addrspace(8)> <ptr addrspace(8) @buf, ptr addrspace(8) null>
  to <2 x ptr addrspace(7)>)
}

; CHECK: %[[#GEP]] = OpFunction %[[#I32PTR_ADDRSPACE_7]]
; CHECK: %[[#BUF_FROM_AS8_TO_AS7:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST_P8_P7]] %[[#BUF]]
; CHECK: %[[#BC_TO_ARR:]] = OpBitcast %[[#ARRPTR_ADDRSPACE_7]] %[[#BUF_FROM_AS8_TO_AS7]]
; CHECK: %[[#GEP_RESULT:]] = OpInBoundsPtrAccessChain %[[#I32PTR_ADDRSPACE_7]] %[[#BC_TO_ARR]]
;	CHECK: OpReturnValue %[[#GEP_RESULT]]
define spir_func ptr addrspace(7) @gep() {
  ret ptr addrspace(7) getelementptr inbounds (
  [4 x i32],
  ptr addrspace(7) addrspacecast (ptr addrspace(8) @buf to ptr addrspace(7)),
  i64 2, i32 1)
}

; CHECK: %[[#GEP_VECTOR]] = OpFunction %[[#VEC2_I8PTR_ADDRSPACE_7]]
; CHECK: %[[#GEP_VEC_RESULT:]] = OpPtrAccessChain %[[#VEC2_I8PTR_ADDRSPACE_7]] %[[#TEMP_VEC_WITH_CONSTEXPR_AS_CAST]]
;	CHECK: OpReturnValue %[[#GEP_VEC_RESULT]]
define spir_func <2 x ptr addrspace(7)> @gep_vector() {
  ret <2 x ptr addrspace(7)> getelementptr (
  <2 x i32>,
  <2 x ptr addrspace(7)>
  <ptr addrspace(7) addrspacecast (ptr addrspace(8) @buf to ptr addrspace(7)),
  ptr addrspace(7) null>,
  <2 x i32> <i32 3, i32 1>)
}

; CHECK: %[[#GEP_OF_P7]] = OpFunction %[[#I8PTR_ADDRSPACE_7PTR_GLOBAL]]
; CHECK: %[[#BC0:]] = OpBitcast %[[#I8PTR_ADDRSPACE_7PTR_GLOBAL]] %[[#FLAT]]
; CHECK: %[[#GEP_P7:]] = OpInBoundsPtrAccessChain %[[#I8PTR_ADDRSPACE_7PTR_GLOBAL]] %[[#BC0]]
; CHECK: OpReturnValue %[[#GEP_P7]]
define spir_func ptr addrspace(1) @gep_of_p7() {
  ret ptr addrspace(1) getelementptr inbounds (ptr addrspace(7), ptr addrspace(1) @flat, i64 2)
}

; CHECK: %[[#GEP_OF_P7_VECTOR]] = OpFunction %[[#VEC2_I8PTR_ADDRSPACE_7PTR_GLOBAL]]
; CHECK: %[[#BC1:]] = OpBitcast %[[#VEC2_I8PTR_ADDRSPACE_7PTR_GLOBAL]] %[[#FLAT]]
; CHECK: %[[#GEP_P7_VEC:]] = OpPtrAccessChain %[[#VEC2_I8PTR_ADDRSPACE_7PTR_GLOBAL]] %[[#BC1]]
; CHECK: OpReturnValue %[[#GEP_P7_VEC]]
define spir_func ptr addrspace(1) @gep_of_p7_vector() {
  ret ptr addrspace(1) getelementptr (<2 x ptr addrspace(7)>, ptr addrspace(1) @flat, i64 2)
}

; CHECK: %[[#GEP_OF_P7_STRUCT]] = OpFunction %[[#STRUCTPTR_GLOBAL]]
; CHECK: %[[#BC2:]] = OpBitcast %[[#STRUCTPTR_GLOBAL]] %[[#FLAT]]
; CHECK: %[[#GEP_P7_STRUCT:]] = OpPtrAccessChain %[[#STRUCTPTR_GLOBAL]] %[[#BC2]]
; CHECK: OpReturnValue %[[#GEP_P7_STRUCT]]
define spir_func ptr addrspace(1) @gep_of_p7_struct() {
  ret ptr addrspace(1) getelementptr ({ptr addrspace(7), i32}, ptr addrspace(1) @flat, i64 2)
}

; CHECK: %[[#GEP_P7_FROM_P7]] = OpFunction %[[#PTRI8PTR_ADDRSPACE_7_ADDRSPACE_7]]
; CHECK: %[[#BUF_AS8_TO_AS7:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST_P8_P7]] %[[#BUF]]
; CHECK: %[[#BC3:]] = OpBitcast %[[#PTRI8PTR_ADDRSPACE_7_ADDRSPACE_7]] %[[#BUF_AS8_TO_AS7]]
; CHECK: %[[#GEP_P7_P7:]] = OpPtrAccessChain %[[#PTRI8PTR_ADDRSPACE_7_ADDRSPACE_7]] %[[#BC3]]
; CHECK: OpReturnValue %[[#GEP_P7_P7]]
define spir_func ptr addrspace(7) @gep_p7_from_p7() {
  ret ptr addrspace(7) getelementptr (ptr addrspace(7),
  ptr addrspace(7) addrspacecast (ptr addrspace(8) @buf to ptr addrspace(7)),
  i64 2)
}

; CHECK: %[[#PTRTOINT]] = OpFunction %[[#INT160_TY]]
; CHECK: %[[#BUF_AS8_TO_AS7_1:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST_P8_P7]] %[[#BUF]]
;	CHECK: %[[#BC4:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#BUF_AS8_TO_AS7_1]]
;	CHECK: %[[#PTR:]] = OpPtrAccessChain %[[#I32PTR_ADDRSPACE_7]] %[[#BC4]]
;	CHECK: %[[#PTR_TO_U160:]] = OpConvertPtrToU %[[#INT160_TY]] %[[#PTR]]
; CHECK: OpReturnValue %[[#PTR_TO_U160]]
define spir_func i160 @ptrtoint() {
  ret i160 ptrtoint(
  ptr addrspace(7) getelementptr(
  i32, ptr addrspace(7) addrspacecast (ptr addrspace(8) @buf to ptr addrspace(7)),
  i32 3) to i160)
}

; CHECK: %[[#PTRTOINT_LONG]] = OpFunction %[[#INT256_TY]]
; CHECK: %[[#BUF_AS8_TO_AS7_2:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST_P8_P7]] %[[#BUF]]
;	CHECK: %[[#BC5:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#BUF_AS8_TO_AS7_2]]
;	CHECK: %[[#PTR1:]] = OpPtrAccessChain %[[#I32PTR_ADDRSPACE_7]] %[[#BC5]]
;	CHECK: %[[#PTR_TO_U256:]] = OpConvertPtrToU %[[#INT256_TY]] %[[#PTR1]]
; CHECK: OpReturnValue %[[#PTR_TO_U256]]
define spir_func i256 @ptrtoint_long() {
  ret i256 ptrtoint(
  ptr addrspace(7) getelementptr(
  i32, ptr addrspace(7) addrspacecast (ptr addrspace(8) @buf to ptr addrspace(7)),
  i32 3) to i256)
}

; CHECK: %[[#PTRTOINT_SHORT]] = OpFunction %[[#INT64_TY]]
; CHECK: %[[#BUF_AS8_TO_AS7_3:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST_P8_P7]] %[[#BUF]]
;	CHECK: %[[#BC6:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#BUF_AS8_TO_AS7_3]]
;	CHECK: %[[#PTR2:]] = OpPtrAccessChain %[[#I32PTR_ADDRSPACE_7]] %[[#BC6]]
;	CHECK: %[[#PTR_TO_U64:]] = OpConvertPtrToU %[[#INT64_TY]] %[[#PTR2]]
; CHECK: OpReturnValue %[[#PTR_TO_U64]]
define spir_func i64 @ptrtoint_short() {
  ret i64 ptrtoint(
  ptr addrspace(7) getelementptr(
  i32, ptr addrspace(7) addrspacecast (ptr addrspace(8) @buf to ptr addrspace(7)),
  i32 3) to i64)
}

; CHECK: %[[#PTRTOINT_VERY_SHORT]] = OpFunction %[[#INT32_TY]]
; CHECK: %[[#BUF_AS8_TO_AS7_4:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST_P8_P7]] %[[#BUF]]
;	CHECK: %[[#BC7:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#BUF_AS8_TO_AS7_4]]
;	CHECK: %[[#PTR3:]] = OpPtrAccessChain %[[#I32PTR_ADDRSPACE_7]] %[[#BC7]]
;	CHECK: %[[#PTR_TO_U32:]] = OpConvertPtrToU %[[#INT32_TY]] %[[#PTR3]]
; CHECK: OpReturnValue %[[#PTR_TO_U32]]
define spir_func i32 @ptrtoint_very_short() {
  ret i32 ptrtoint(
  ptr addrspace(7) getelementptr(
  i32, ptr addrspace(7) addrspacecast (ptr addrspace(8) @buf to ptr addrspace(7)),
  i32 3) to i32)
}

; CHECK: %[[#PTRTOINT_VEC]] = OpFunction %[[#VEC2_I160_TY]]
; CHECK: OpReturnValue %[[#NULL_VEC2_I160]]
define spir_func <2 x i160> @ptrtoint_vec() {
  ret <2 x i160> ptrtoint (<2 x ptr addrspace(7)> zeroinitializer to <2 x i160>)
}

; CHECK: %[[#INTTOPTR]] = OpFunction %[[#I8PTR_ADDRSPACE_7]]
; CHECK: OpReturnValue %[[#NULLPTR_I8PTR_ADDRSPACE_7]]
define spir_func ptr addrspace(7) @inttoptr() {
  ret ptr addrspace(7) inttoptr (i160 0 to ptr addrspace(7))
}

; CHECK: %[[#INTTOPTR_VEC]] = OpFunction %[[#VEC2_I8PTR_ADDRSPACE_7]]
; CHECK: %[[#TMPV:]] = OpCompositeInsert %[[#VEC2_I8PTR_ADDRSPACE_7]] %[[#CONSTEXPR_CAST_INT_1_TO_PTR_AS7]] %[[#]] 0
; CHECK: %[[#V:]] = OpCompositeInsert %[[#VEC2_I8PTR_ADDRSPACE_7]] %[[#CONSTEXPR_CAST_INT_2_TO_PTR_AS7]] %[[#TMPV]] 1
;	OpReturnValue %[[#V]]
define spir_func <2 x ptr addrspace(7)> @inttoptr_vec() {
  ret <2 x ptr addrspace(7)> inttoptr (<2 x i160> <i160 1, i160 2> to <2 x ptr addrspace(7)>)
}

; CHECK: %[[#FANCY_ZERO]] = OpFunction %[[#INT32_TY]]
; CHECK: %[[#BUF_AS8_TO_AS7_5:]] = OpFunctionCall %[[#I8PTR_ADDRSPACE_7]] %[[#OPAQUE_PTR_CAST_P8_P7]] %[[#BUF]]
; CHECK: %[[#PTR_TO_U32_1:]] = OpConvertPtrToU %[[#INT32_TY]] %[[#BUF_AS8_TO_AS7_5]]
; CHECK: OpReturnValue %[[#PTR_TO_U32_1]]
define spir_func i32 @fancy_zero() {
  ret i32 ptrtoint (
  ptr addrspace(7) addrspacecast (ptr addrspace(8) @buf to ptr addrspace(7))
  to i32)
}

; CHECK: %[[#LOAD_NULL]] = OpFunction %[[#INT32_TY]]
;	CHECK: %[[#BC8:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#NULLPTR_I8PTR_ADDRSPACE_7]]
;	CHECK: %[[#LD:]] = OpLoad %[[#INT32_TY]] %[[#BC8]]
; CHECK: OpReturnValue %[[#LD]]
define spir_func i32 @load_null() {
  %x = load i32, ptr addrspace(7) null, align 4
  ret i32 %x
}

; CHECK: %[[#STORE_NULL]] = OpFunction
;	CHECK: %[[#BC9:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#NULLPTR_I8PTR_ADDRSPACE_7]]
;	CHECK: OpStore %[[#BC9]]
define spir_func void @store_null() {
  store i32 0, ptr addrspace(7) null, align 4
  ret void
}

; CHECK: %[[#LOAD_POISON]] = OpFunction %[[#INT32_TY]]
;	CHECK: %[[#BC10:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#UNDEF_I8PTR_ADDRSPACE_7:]]
;	CHECK: %[[#LD1:]] = OpLoad %[[#INT32_TY]] %[[#BC10]]
; CHECK: OpReturnValue %[[#LD1]]
define spir_func i32 @load_poison() {
  %x = load i32, ptr addrspace(7) poison, align 4
  ret i32 %x
}

; CHECK: %[[#STORE_POISON]] = OpFunction
;	CHECK: %[[#BC11:]] = OpBitcast %[[#I32PTR_ADDRSPACE_7]] %[[#UNDEF_I8PTR_ADDRSPACE_7:]]
;	CHECK: OpStore %[[#BC11]]
define spir_func void @store_poison() {
  store i32 0, ptr addrspace(7) poison, align 4
  ret void
}
