; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_long_vector,+SPV_INTEL_masked_gather_scatter %s -o - | FileCheck %s
; spirv-val seems to have problems reading OpTypeVectorIdEXT correctly, enable once fixed
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
; CHECK-DAG: %[[#VPTR17:]] = OpTypeVectorIdEXT %[[#PTR]] 17
; CHECK-DAG: %[[#VPTR1:]] = OpTypeVectorIdEXT %[[#PTR]] 1
; CHECK-DAG: %[[#VI64_17:]] = OpTypeVectorIdEXT %[[#I64]] 17
; CHECK-DAG: %[[#UNDEF1:]] = OpUndef %[[#VPTR1]]
; CHECK-DAG: %[[#UNDEF17:]] = OpUndef %[[#VPTR17]]
; CHECK-DAG: %[[#NULL17:]] = OpConstantNull %[[#VI64_17]]

; CHECK:      OpFunction
; CHECK-NEXT: %[[#P1:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: %[[#OUT1:]] = OpFunctionParameter %[[#PTR]]
; CHECK-NEXT: OpLabel
; CHECK-NEXT: %[[#GEP1:]] = OpPtrAccessChain %[[#PTR]] %[[#P1]] %[[#FIVE]]
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
; CHECK-NEXT: %[[#GEPV_0:]] = OpPtrAccessChain %[[#PTR]] %[[#EXPV_0]] %[[#ONE]]
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
; CHECK-NEXT: %[[#GEP0:]] = OpPtrAccessChain %[[#PTR]] %[[#P0]] %[[#ZERO]]
; CHECK-NEXT: %[[#TMP0:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP0]] %[[#UNDEF17]] 0
; CHECK-NEXT: %[[#P1:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 1
; CHECK-NEXT: %[[#GEP1:]] = OpPtrAccessChain %[[#PTR]] %[[#P1]] %[[#ONE]]
; CHECK-NEXT: %[[#TMP1:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP1]] %[[#TMP0]] 1
; CHECK-NEXT: %[[#P2:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 2
; CHECK-NEXT: %[[#GEP2:]] = OpPtrAccessChain %[[#PTR]] %[[#P2]] %[[#TWO]]
; CHECK-NEXT: %[[#TMP2:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP2]] %[[#TMP1]] 2
; CHECK-NEXT: %[[#P3:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 3
; CHECK-NEXT: %[[#GEP3:]] = OpPtrAccessChain %[[#PTR]] %[[#P3]] %[[#THREE]]
; CHECK-NEXT: %[[#TMP3:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP3]] %[[#TMP2]] 3
; CHECK-NEXT: %[[#P4:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 4
; CHECK-NEXT: %[[#GEP4:]] = OpPtrAccessChain %[[#PTR]] %[[#P4]] %[[#FOUR]]
; CHECK-NEXT: %[[#TMP4:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP4]] %[[#TMP3]] 4
; CHECK-NEXT: %[[#P5:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 5
; CHECK-NEXT: %[[#GEP5:]] = OpPtrAccessChain %[[#PTR]] %[[#P5]] %[[#FIVE]]
; CHECK-NEXT: %[[#TMP5:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP5]] %[[#TMP4]] 5
; CHECK-NEXT: %[[#P6:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 6
; CHECK-NEXT: %[[#GEP6:]] = OpPtrAccessChain %[[#PTR]] %[[#P6]] %[[#SIX]]
; CHECK-NEXT: %[[#TMP6:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP6]] %[[#TMP5]] 6
; CHECK-NEXT: %[[#P7:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 7
; CHECK-NEXT: %[[#GEP7:]] = OpPtrAccessChain %[[#PTR]] %[[#P7]] %[[#SEVEN]]
; CHECK-NEXT: %[[#TMP7:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP7]] %[[#TMP6]] 7
; CHECK-NEXT: %[[#P8:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 8
; CHECK-NEXT: %[[#GEP8:]] = OpPtrAccessChain %[[#PTR]] %[[#P8]] %[[#EIGHT]]
; CHECK-NEXT: %[[#TMP8:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP8]] %[[#TMP7]] 8
; CHECK-NEXT: %[[#P9:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 9
; CHECK-NEXT: %[[#GEP9:]] = OpPtrAccessChain %[[#PTR]] %[[#P9]] %[[#NINE]]
; CHECK-NEXT: %[[#TMP9:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP9]] %[[#TMP8]] 9
; CHECK-NEXT: %[[#P10:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 10
; CHECK-NEXT: %[[#GEP10:]] = OpPtrAccessChain %[[#PTR]] %[[#P10]] %[[#TEN]]
; CHECK-NEXT: %[[#TMP10:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP10]] %[[#TMP9]] 10
; CHECK-NEXT: %[[#P11:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 11
; CHECK-NEXT: %[[#GEP11:]] = OpPtrAccessChain %[[#PTR]] %[[#P11]] %[[#ELEVEN]]
; CHECK-NEXT: %[[#TMP11:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP11]] %[[#TMP10]] 11
; CHECK-NEXT: %[[#P12:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 12
; CHECK-NEXT: %[[#GEP12:]] = OpPtrAccessChain %[[#PTR]] %[[#P12]] %[[#TWELVE]]
; CHECK-NEXT: %[[#TMP12:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP12]] %[[#TMP11]] 12
; CHECK-NEXT: %[[#P13:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 13
; CHECK-NEXT: %[[#GEP13:]] = OpPtrAccessChain %[[#PTR]] %[[#P13]] %[[#THIRTEEN]]
; CHECK-NEXT: %[[#TMP13:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP13]] %[[#TMP12]] 13
; CHECK-NEXT: %[[#P14:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 14
; CHECK-NEXT: %[[#GEP14:]] = OpPtrAccessChain %[[#PTR]] %[[#P14]] %[[#FOURTEEN]]
; CHECK-NEXT: %[[#TMP14:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP14]] %[[#TMP13]] 14
; CHECK-NEXT: %[[#P15:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 15
; CHECK-NEXT: %[[#GEP15:]] = OpPtrAccessChain %[[#PTR]] %[[#P15]] %[[#FIFTEEN]]
; CHECK-NEXT: %[[#TMP15:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP15]] %[[#TMP14]] 15
; CHECK-NEXT: %[[#P16:]] = OpCompositeExtract %[[#PTR]] %[[#PV]] 16
; CHECK-NEXT: %[[#GEP16:]] = OpPtrAccessChain %[[#PTR]] %[[#P16]] %[[#SIXTEEN]]
; CHECK-NEXT: %[[#TMP16:]] = OpCompositeInsert %[[#VPTR17]] %[[#GEP16]] %[[#TMP15]] 16
; CHECK-NEXT: %[[#EXTRACT:]] = OpCompositeExtract %[[#PTR]] %[[#TMP16]] 0
; CHECK-NEXT: %[[#LD:]] = OpLoad %[[#I32]] %[[#EXTRACT]] Aligned 4
; CHECK-NEXT: OpStore %[[#OUTV]] %[[#LD]] Aligned 4
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd
	; %105 = OpCompositeExtract %3 %102 0
	; %106 = OpPtrAccessChain %3 %105 %14
	; %107 = OpCompositeInsert %9 %106 %11 0
	; %108 = OpCompositeExtract %3 %102 1
	; %109 = OpPtrAccessChain %3 %108 %30
	; %110 = OpCompositeInsert %9 %109 %107 1
	; %111 = OpCompositeExtract %3 %102 2
	; %112 = OpPtrAccessChain %3 %111 %29
	; %113 = OpCompositeInsert %9 %112 %110 2
	; %114 = OpCompositeExtract %3 %102 3
	; %115 = OpPtrAccessChain %3 %114 %28
	; %116 = OpCompositeInsert %9 %115 %113 3
	; %117 = OpCompositeExtract %3 %102 4
	; %118 = OpPtrAccessChain %3 %117 %8
	; %119 = OpCompositeInsert %9 %118 %116 4
	; %120 = OpCompositeExtract %3 %102 5
	; %121 = OpPtrAccessChain %3 %120 %27
	; %122 = OpCompositeInsert %9 %121 %119 5
	; %123 = OpCompositeExtract %3 %102 6
	; %124 = OpPtrAccessChain %3 %123 %26
	; %125 = OpCompositeInsert %9 %124 %122 6
	; %126 = OpCompositeExtract %3 %102 7
	; %127 = OpPtrAccessChain %3 %126 %25
	; %128 = OpCompositeInsert %9 %127 %125 7
	; %129 = OpCompositeExtract %3 %102 8
	; %130 = OpPtrAccessChain %3 %129 %24
	; %131 = OpCompositeInsert %9 %130 %128 8
	; %132 = OpCompositeExtract %3 %102 9
	; %133 = OpPtrAccessChain %3 %132 %23
	; %134 = OpCompositeInsert %9 %133 %131 9
	; %135 = OpCompositeExtract %3 %102 10
	; %136 = OpPtrAccessChain %3 %135 %22
	; %137 = OpCompositeInsert %9 %136 %134 10
	; %138 = OpCompositeExtract %3 %102 11
	; %139 = OpPtrAccessChain %3 %138 %21
	; %140 = OpCompositeInsert %9 %139 %137 11
	; %141 = OpCompositeExtract %3 %102 12
	; %142 = OpPtrAccessChain %3 %141 %20
	; %143 = OpCompositeInsert %9 %142 %140 12
	; %144 = OpCompositeExtract %3 %102 13
	; %145 = OpPtrAccessChain %3 %144 %19
	; %146 = OpCompositeInsert %9 %145 %143 13
	; %147 = OpCompositeExtract %3 %102 14
	; %148 = OpPtrAccessChain %3 %147 %18
	; %149 = OpCompositeInsert %9 %148 %146 14
	; %150 = OpCompositeExtract %3 %102 15
	; %151 = OpPtrAccessChain %3 %150 %17
	; %152 = OpCompositeInsert %9 %151 %149 15
	; %153 = OpCompositeExtract %3 %102 16
	; %154 = OpPtrAccessChain %3 %153 %16
	; %155 = OpCompositeInsert %9 %154 %152 16
	; %156 = OpCompositeExtract %3 %155 0
	; %157 = OpLoad %2 %156 Aligned 4
	; OpStore %103 %157 Aligned 4
define spir_kernel void @test_vector_gep_vec_ptr(<17 x ptr addrspace(1)> %ptrs, ptr addrspace(1) %out) {
  %gep = getelementptr i32, <17 x ptr addrspace(1)> %ptrs, <17 x i64> <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15, i64 16>
  %elem = extractelement <17 x ptr addrspace(1)> %gep, i32 0
  %val = load i32, ptr addrspace(1) %elem
  store i32 %val, ptr addrspace(1) %out
  ret void
}
