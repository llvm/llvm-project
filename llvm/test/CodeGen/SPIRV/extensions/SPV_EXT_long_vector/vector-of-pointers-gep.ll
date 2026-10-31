; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector,+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; spirv-val has a bug around validating OpTypeVectorIdEXTs with Pointer type elements, as it seems to only allow scalar numerical types for the latter.
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector,+SPV_INTEL_masked_gather_scatter %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#PTR:]] = OpTypePointer CrossWorkgroup %[[#I32]]
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#ZERO:]] = OpConstantNull %[[#I64]]
; CHECK-DAG: %[[#ONE:]] = OpConstant %[[#I64]] 1
; CHECK-DAG: %[[#TWO:]] = OpConstant %[[#I64]] 2
; CHECK-DAG: %[[#THREE:]] = OpConstant %[[#I64]] 3
; CHECK-DAG: %[[#FOUR:]] = OpConstant %[[#I64]] 4
; CHECK-DAG: %[[#FIVE:]] = OpConstant %[[#I64]] 5
; CHECK-DAG: %[[#SIX:]] = OpConstant %[[#I64]] 6
; CHECK-DAG: %[[#SEVEN:]] = OpConstant %[[#I64]] 7
; CHECK-DAG: %[[#EIGHT:]] = OpConstant %[[#I64]] 8
; CHECK-DAG: %[[#NINE:]] = OpConstant %[[#I64]] 9
; CHECK-DAG: %[[#TEN:]] = OpConstant %[[#I64]] 10
; CHECK-DAG: %[[#ELEVEN:]] = OpConstant %[[#I64]] 11
; CHECK-DAG: %[[#TWELVE:]] = OpConstant %[[#I64]] 12
; CHECK-DAG: %[[#THIRTEEN:]] = OpConstant %[[#I64]] 13
; CHECK-DAG: %[[#FOURTEEN:]] = OpConstant %[[#I64]] 14
; CHECK-DAG: %[[#FIFTEEN:]] = OpConstant %[[#I64]] 15
; CHECK-DAG: %[[#SIXTEEN:]] = OpConstant %[[#I64]] 16
; CHECK-DAG: %[[#SEVENTEEN:]] = OpConstant %[[#I32]] 17
; CHECK-DAG: %[[#ONE_I32:]] = OpConstant %[[#I32]] 1
; CHECK-DAG: %[[#VPTR17:]] = OpTypeVectorIdEXT %[[#PTR]] %[[#SEVENTEEN]]
; CHECK-DAG: %[[#VPTR1:]] = OpTypeVectorIdEXT %[[#PTR]] %[[#ONE_I32]]
; CHECK-DAG: %[[#VI64_1:]] = OpTypeVectorIdEXT %[[#I64]] %[[#ONE_I32]]
; CHECK-DAG: %[[#VI64_17:]] = OpTypeVectorIdEXT %[[#I64]] %[[#SEVENTEEN]]
; CHECK-DAG: %[[#UNDEF1:]] = OpUndef %[[#VPTR1]]
; CHECK-DAG: %[[#UNDEF17:]] = OpUndef %[[#VPTR17]]
; CHECK-DAG: %[[#NULL17:]] = OpConstantNull %[[#VI64_17]]
; CHECK-DAG: %[[#ONE_V1:]] = OpConstantComposite %[[#VI64_1]] %[[#ONE]]
; CHECK-DAG: %[[#FIVE_V1:]] = OpConstantComposite %[[#VI64_1]] %[[#FIVE]]
; CHECK-DAG: %[[#IDXS_V17:]] = OpConstantComposite %[[#VI64_17]] %[[#ZERO]] %[[#ONE]] %[[#TWO]] %[[#THREE]] %[[#FOUR]] %[[#FIVE]] %[[#SIX]] %[[#SEVEN]] %[[#EIGHT]] %[[#NINE]] %[[#TEN]] %[[#ELEVEN]] %[[#TWELVE]] %[[#THIRTEEN]] %[[#FOURTEEN]] %[[#FIFTEEN]] %[[#SIXTEEN]]

; CHECK:      OpFunction
; CHECK-NEXT: %[[#P1:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: %[[#OUT1:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#IDX1:]] = OpCompositeExtract %[[#I64]] %[[#FIVE_V1]]
; CHECK-NEXT: %[[#GEP1:]] = OpPtrAccessChain %[[#PTR]] %[[#P1]] %[[#IDX1]]
; CHECK-NEXT: %[[#INSERT:]] = OpCompositeInsert %[[#VPTR1]] %[[#GEP1]] %[[#UNDEF1]] 0
; CHECK-NEXT: %[[#EXTRACT:]] = OpCompositeExtract %[[#PTR]] %[[#INSERT]] 0
; CHECK-NEXT: %[[#VAL1:]] = OpLoad %[[#I32]] %[[#EXTRACT]]
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
; CHECK-NEXT: %[[#IDX0:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 0
; CHECK-NEXT: %[[#GEP0:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX0]]
; CHECK-NEXT: %[[#TMP0:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP0]] %[[#UNDEF17]] 0
; CHECK-NEXT: %[[#IDX1:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 1
; CHECK-NEXT: %[[#GEP1:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX1]]
; CHECK-NEXT: %[[#TMP1:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP1]] %[[#TMP0]] 1
; CHECK-NEXT: %[[#IDX2:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 2
; CHECK-NEXT: %[[#GEP2:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX2]]
; CHECK-NEXT: %[[#TMP2:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP2]] %[[#TMP1]] 2
; CHECK-NEXT: %[[#IDX3:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 3
; CHECK-NEXT: %[[#GEP3:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX3]]
; CHECK-NEXT: %[[#TMP3:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP3]] %[[#TMP2]] 3
; CHECK-NEXT: %[[#IDX4:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 4
; CHECK-NEXT: %[[#GEP4:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX4]]
; CHECK-NEXT: %[[#TMP4:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP4]] %[[#TMP3]] 4
; CHECK-NEXT: %[[#IDX5:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 5
; CHECK-NEXT: %[[#GEP5:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX5]]
; CHECK-NEXT: %[[#TMP5:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP5]] %[[#TMP4]] 5
; CHECK-NEXT: %[[#IDX6:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 6
; CHECK-NEXT: %[[#GEP6:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX6]]
; CHECK-NEXT: %[[#TMP6:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP6]] %[[#TMP5]] 6
; CHECK-NEXT: %[[#IDX7:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 7
; CHECK-NEXT: %[[#GEP7:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX7]]
; CHECK-NEXT: %[[#TMP7:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP7]] %[[#TMP6]] 7
; CHECK-NEXT: %[[#IDX8:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 8
; CHECK-NEXT: %[[#GEP8:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX8]]
; CHECK-NEXT: %[[#TMP8:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP8]] %[[#TMP7]] 8
; CHECK-NEXT: %[[#IDX9:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 9
; CHECK-NEXT: %[[#GEP9:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX9]]
; CHECK-NEXT: %[[#TMP9:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP9]] %[[#TMP8]] 9
; CHECK-NEXT: %[[#IDX10:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 10
; CHECK-NEXT: %[[#GEP10:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX10]]
; CHECK-NEXT: %[[#TMP10:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP10]] %[[#TMP9]] 10
; CHECK-NEXT: %[[#IDX11:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 11
; CHECK-NEXT: %[[#GEP11:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX11]]
; CHECK-NEXT: %[[#TMP11:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP11]] %[[#TMP10]] 11
; CHECK-NEXT: %[[#IDX12:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 12
; CHECK-NEXT: %[[#GEP12:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX12]]
; CHECK-NEXT: %[[#TMP12:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP12]] %[[#TMP11]] 12
; CHECK-NEXT: %[[#IDX13:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 13
; CHECK-NEXT: %[[#GEP13:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX13]]
; CHECK-NEXT: %[[#TMP13:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP13]] %[[#TMP12]] 13
; CHECK-NEXT: %[[#IDX14:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 14
; CHECK-NEXT: %[[#GEP14:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX14]]
; CHECK-NEXT: %[[#TMP14:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP14]] %[[#TMP13]] 14
; CHECK-NEXT: %[[#IDX15:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 15
; CHECK-NEXT: %[[#GEP15:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX15]]
; CHECK-NEXT: %[[#TMP15:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP15]] %[[#TMP14]] 15
; CHECK-NEXT: %[[#IDX16:]] = OpCompositeExtract %[[#I64]] %[[#NULL17]] 16
; CHECK-NEXT: %[[#GEP16:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX16]]
; CHECK-NEXT: %[[#TMP16:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP16]] %[[#TMP15]] 16
; CHECK-NEXT: %[[#EXTRACT:]] = OpCompositeExtract %[[#PTR]] %[[#TMP16]] 0
; CHECK-NEXT: %[[#LD:]] = OpLoad %[[#I32]] %[[#EXTRACT]] Aligned 4
; CHECK-NEXT: OpStore %[[#OUT2]] %[[#LD]] Aligned 4
; CHECK-NEXT: OpReturn
; CHECK: OpFunctionEnd
define spir_kernel void @test_vector_gep_v17(ptr addrspace(1) %p, ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %p, <17 x i64> zeroinitializer
  %elem = extractelement <17 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK:      OpFunction
; CHECK-NEXT: %[[#PV:]] = OpFunctionParameter %[[#VPTR1]]
; CHECK-NEXT: %[[#OUTV:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#EXPV_0:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 0
; CHECK-NEXT: %[[#IDXV_0:]] = OpCompositeExtract %[[#I64]] %[[#ONE_V1]] 0
; CHECK-NEXT: %[[#GEPV_0:]] = OpPtrAccessChain %[[#PTR]] %[[#EXPV_0]] %[[#IDXV_0]]
; CHECK-NEXT: %[[#INSV_0:]] = OpCompositeInsert %[[#VPTR1]] %[[#GEPV_0]] %[[#UNDEF1]] 0
; CHECK-NEXT: %[[#EXPV_1:]] = OpCompositeExtract %[[#PTR]] %[[#INSV_0]] 0
; CHECK-NEXT: %[[#VALV:]] = OpLoad %[[#I32]] %[[#EXPV_1]]
; CHECK-NEXT: OpStore %[[#OUTV]] %[[#VALV]]
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
define spir_kernel void @test_vector_gep_vec1_ptr(<1 x ptr addrspace(1)> %ptrs, ptr addrspace(1) %out) {
  %gep = getelementptr i32, <1 x ptr addrspace(1)> %ptrs, <1 x i64> <i64 1>
  %elem = extractelement <1 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; CHECK:      OpFunction
; CHECK-NEXT: %[[#PV:]] = OpFunctionParameter %[[#VPTR17]]
; CHECK-NEXT: %[[#OUTV:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#P0:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 0
; CHECK-NEXT: %[[#IDX0:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 0
; CHECK-NEXT: %[[#GEP0:]] = OpPtrAccessChain %[[#PTR]] %[[#P0]] %[[#IDX0]]
; CHECK-NEXT: %[[#TMP0:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP0]] %[[#UNDEF17]] 0
; CHECK-NEXT: %[[#P1:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 1
; CHECK-NEXT: %[[#IDX1:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 1
; CHECK-NEXT: %[[#GEP1:]] = OpPtrAccessChain %[[#PTR]] %[[#P1]] %[[#IDX1]]
; CHECK-NEXT: %[[#TMP1:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP1]] %[[#TMP0]] 1
; CHECK-NEXT: %[[#P2:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 2
; CHECK-NEXT: %[[#IDX2:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 2
; CHECK-NEXT: %[[#GEP2:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#IDX2]]
; CHECK-NEXT: %[[#TMP2:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP2]] %[[#TMP1]] 2
; CHECK-NEXT: %[[#P3:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 3
; CHECK-NEXT: %[[#IDX3:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 3
; CHECK-NEXT: %[[#GEP3:]] = OpPtrAccessChain %[[#PTR]] %[[#P3]] %[[#IDX3]]
; CHECK-NEXT: %[[#TMP3:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP3]] %[[#TMP2]] 3
; CHECK-NEXT: %[[#P4:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 4
; CHECK-NEXT: %[[#IDX4:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 4
; CHECK-NEXT: %[[#GEP4:]] = OpPtrAccessChain %[[#PTR]] %[[#P4]] %[[#IDX4]]
; CHECK-NEXT: %[[#TMP4:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP4]] %[[#TMP3]] 4
; CHECK-NEXT: %[[#P5:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 5
; CHECK-NEXT: %[[#IDX5:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 5
; CHECK-NEXT: %[[#GEP5:]] = OpPtrAccessChain %[[#PTR]] %[[#P5]] %[[#IDX5]]
; CHECK-NEXT: %[[#TMP5:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP5]] %[[#TMP4]] 5
; CHECK-NEXT: %[[#P6:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 6
; CHECK-NEXT: %[[#IDX6:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 6
; CHECK-NEXT: %[[#GEP6:]] = OpPtrAccessChain %[[#PTR]] %[[#P6]] %[[#IDX6]]
; CHECK-NEXT: %[[#TMP6:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP6]] %[[#TMP5]] 6
; CHECK-NEXT: %[[#P7:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 7
; CHECK-NEXT: %[[#IDX7:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 7
; CHECK-NEXT: %[[#GEP7:]] = OpPtrAccessChain %[[#PTR]] %[[#P7]] %[[#IDX7]]
; CHECK-NEXT: %[[#TMP7:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP7]] %[[#TMP6]] 7
; CHECK-NEXT: %[[#P8:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 8
; CHECK-NEXT: %[[#IDX8:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 8
; CHECK-NEXT: %[[#GEP8:]] = OpPtrAccessChain %[[#PTR]] %[[#P8]] %[[#IDX8]]
; CHECK-NEXT: %[[#TMP8:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP8]] %[[#TMP7]] 8
; CHECK-NEXT: %[[#P9:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 9
; CHECK-NEXT: %[[#IDX9:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 9
; CHECK-NEXT: %[[#GEP9:]] = OpPtrAccessChain %[[#PTR]] %[[#P9]] %[[#IDX9]]
; CHECK-NEXT: %[[#TMP9:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP9]] %[[#TMP8]] 9
; CHECK-NEXT: %[[#P10:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 10
; CHECK-NEXT: %[[#IDX10:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 10
; CHECK-NEXT: %[[#GEP10:]] = OpPtrAccessChain %[[#PTR]] %[[#P10]] %[[#IDX10]]
; CHECK-NEXT: %[[#TMP10:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP10]] %[[#TMP9]] 10
; CHECK-NEXT: %[[#P11:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 11
; CHECK-NEXT: %[[#IDX11:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 11
; CHECK-NEXT: %[[#GEP11:]] = OpPtrAccessChain %[[#PTR]] %[[#P11]] %[[#IDX11]]
; CHECK-NEXT: %[[#TMP11:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP11]] %[[#TMP10]] 11
; CHECK-NEXT: %[[#P12:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 12
; CHECK-NEXT: %[[#IDX12:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 12
; CHECK-NEXT: %[[#GEP12:]] = OpPtrAccessChain %[[#PTR]] %[[#P12]] %[[#IDX12]]
; CHECK-NEXT: %[[#TMP12:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP12]] %[[#TMP11]] 12
; CHECK-NEXT: %[[#P13:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 13
; CHECK-NEXT: %[[#IDX13:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 13
; CHECK-NEXT: %[[#GEP13:]] = OpPtrAccessChain %[[#PTR]] %[[#P13]] %[[#IDX13]]
; CHECK-NEXT: %[[#TMP13:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP13]] %[[#TMP12]] 13
; CHECK-NEXT: %[[#P14:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 14
; CHECK-NEXT: %[[#IDX14:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 14
; CHECK-NEXT: %[[#GEP14:]] = OpPtrAccessChain %[[#PTR]] %[[#P14]] %[[#IDX14]]
; CHECK-NEXT: %[[#TMP14:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP14]] %[[#TMP13]] 14
; CHECK-NEXT: %[[#P15:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 15
; CHECK-NEXT: %[[#IDX15:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 15
; CHECK-NEXT: %[[#GEP15:]] = OpPtrAccessChain %[[#PTR]] %[[#P15]] %[[#IDX15]]
; CHECK-NEXT: %[[#TMP15:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP15]] %[[#TMP14]] 15
; CHECK-NEXT: %[[#P16:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 16
; CHECK-NEXT: %[[#IDX16:]] = OpCompositeExtract %[[#I64]] %[[#IDXS_V17]] 16
; CHECK-NEXT: %[[#GEP16:]] = OpPtrAccessChain %[[#PTR]] %[[#P16]] %[[#IDX16]]
; CHECK-NEXT: %[[#TMP16:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP16]] %[[#TMP15]] 16
; CHECK-NEXT: %[[#EXTRACT:]] = OpCompositeExtract %[[#PTR]] %[[#TMP16]] 0
; CHECK-NEXT: %[[#LD:]] = OpLoad %[[#I32]] %[[#EXTRACT]] Aligned 4
; CHECK-NEXT: OpStore %[[#OUTV]] %[[#LD]] Aligned 4
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
define spir_kernel void @test_vector_gep_vec_ptr(<17 x ptr addrspace(1)> %ptrs, ptr addrspace(1) %out) {
  %gep = getelementptr i32, <17 x ptr addrspace(1)> %ptrs, <17 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16>
  %elem = extractelement <17 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}
