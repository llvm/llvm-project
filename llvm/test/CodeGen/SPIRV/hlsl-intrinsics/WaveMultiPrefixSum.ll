; DEFINE: %{nv} = SPV_NV_shader_subgroup_partitioned
; DEFINE: %{ext} = SPV_EXT_shader_subgroup_partitioned
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown \
; RUN:   --spirv-ext=+%{nv} %s -o - | FileCheck %s -DEXTENSION=NV \
; RUN:   --implicit-check-not=%{ext}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown \
; RUN:   --spirv-ext=+%{ext} %s -o - | FileCheck %s -DEXTENSION=EXT \
; RUN:   --implicit-check-not=%{nv}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown \
; RUN:   --spirv-ext=+%{nv},+%{ext} %s -o - | FileCheck %s -DEXTENSION=EXT \
; RUN:   --implicit-check-not=%{nv}
; RUN: not llc -O0 -mtriple=spirv-vulkan-unknown %s -o - 2>&1 | \
; RUN:   FileCheck %s --check-prefix=NO-EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown \
; RUN:   --spirv-ext=+%{nv} %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown \
; RUN:   --spirv-ext=+%{ext} %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown \
; RUN:   --spirv-ext=+%{nv},+%{ext} %s -o - \
; RUN:   -filetype=obj | spirv-val %}

; NO-EXT: WaveMultiPrefixSum requires SPV_EXT_shader_subgroup_partitioned or SPV_NV_shader_subgroup_partitioned

; CHECK-DAG: OpCapability GroupNonUniformPartitionedNV
; CHECK-DAG: OpExtension "SPV_[[EXTENSION]]_shader_subgroup_partitioned"
; CHECK-DAG: %[[#I16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#F16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#F64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#V2I32:]] = OpTypeVector %[[#I32]] 2
; CHECK-DAG: %[[#V4I32:]] = OpTypeVector %[[#I32]] 4
; CHECK-DAG: %[[#V4F32:]] = OpTypeVector %[[#F32]] 4
; CHECK-DAG: %[[#SCOPE:]] = OpConstant %[[#I32]] 3
; CHECK-DAG: %[[#C1:]] = OpConstant %[[#I32]] 1
; CHECK-DAG: %[[#C2:]] = OpConstant %[[#I32]] 2
; CHECK-DAG: %[[#C4:]] = OpConstant %[[#I32]] 4
; CHECK-DAG: %[[#C8:]] = OpConstant %[[#I32]] 8

@wide_float = internal addrspace(10) global [5 x float] zeroinitializer

; CHECK-LABEL: Begin function test_float
; CHECK: %[[#VALUE:]] = OpFunctionParameter %[[#F32]]
; CHECK: %[[#M0:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#M1:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#M2:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#M3:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#MASK:]] = OpCompositeConstruct %[[#V4I32]] %[[#M0]] %[[#M1]] %[[#M2]] %[[#M3]]
; CHECK: %[[#RET:]] = OpGroupNonUniformFAdd %[[#F32]] %[[#SCOPE]] PartitionedExclusiveScanNV %[[#VALUE]] %[[#MASK]]
; CHECK: OpReturnValue %[[#RET]]
define float @test_float(float %value, i32 %m0, i32 %m1,
                         i32 %m2, i32 %m3) {
  %ret = call float @llvm.spv.subgroup.multi.prefix.sum.f32(
      float %value, i32 %m0, i32 %m1, i32 %m2, i32 %m3)
  ret float %ret
}

; CHECK-LABEL: Begin function test_half
; CHECK: %[[#VALUE:]] = OpFunctionParameter %[[#F16]]
; CHECK: %[[#MASK:]] = OpCompositeConstruct %[[#V4I32]] %[[#C1]] %[[#C2]] %[[#C4]] %[[#C8]]
; CHECK: %[[#RET:]] = OpGroupNonUniformFAdd %[[#F16]] %[[#SCOPE]] PartitionedExclusiveScanNV %[[#VALUE]] %[[#MASK]]
; CHECK: OpReturnValue %[[#RET]]
define half @test_half(half %value) {
  %ret = call half @llvm.spv.subgroup.multi.prefix.sum.f16(
      half %value, i32 1, i32 2, i32 4, i32 8)
  ret half %ret
}

; CHECK-LABEL: Begin function test_double
; CHECK: %[[#VALUE:]] = OpFunctionParameter %[[#F64]]
; CHECK: %[[#MASK:]] = OpCompositeConstruct %[[#V4I32]] %[[#C1]] %[[#C2]] %[[#C4]] %[[#C8]]
; CHECK: %[[#RET:]] = OpGroupNonUniformFAdd %[[#F64]] %[[#SCOPE]] PartitionedExclusiveScanNV %[[#VALUE]] %[[#MASK]]
; CHECK: OpReturnValue %[[#RET]]
define double @test_double(double %value) {
  %ret = call double @llvm.spv.subgroup.multi.prefix.sum.f64(
      double %value, i32 1, i32 2, i32 4, i32 8)
  ret double %ret
}

; CHECK-LABEL: Begin function test_i16
; CHECK: %[[#VALUE:]] = OpFunctionParameter %[[#I16]]
; CHECK: %[[#MASK:]] = OpCompositeConstruct %[[#V4I32]] %[[#C1]] %[[#C2]] %[[#C4]] %[[#C8]]
; CHECK: %[[#RET:]] = OpGroupNonUniformIAdd %[[#I16]] %[[#SCOPE]] PartitionedExclusiveScanNV %[[#VALUE]] %[[#MASK]]
; CHECK: OpReturnValue %[[#RET]]
define i16 @test_i16(i16 %value) {
  %ret = call i16 @llvm.spv.subgroup.multi.prefix.sum.i16(
      i16 %value, i32 1, i32 2, i32 4, i32 8)
  ret i16 %ret
}

; CHECK-LABEL: Begin function test_i32
; CHECK: %[[#VALUE:]] = OpFunctionParameter %[[#I32]]
; CHECK: %[[#MASK:]] = OpCompositeConstruct %[[#V4I32]] %[[#C1]] %[[#C2]] %[[#C4]] %[[#C8]]
; CHECK: %[[#RET:]] = OpGroupNonUniformIAdd %[[#I32]] %[[#SCOPE]] PartitionedExclusiveScanNV %[[#VALUE]] %[[#MASK]]
; CHECK: OpReturnValue %[[#RET]]
define i32 @test_i32(i32 %value) {
  %ret = call i32 @llvm.spv.subgroup.multi.prefix.sum.i32(
      i32 %value, i32 1, i32 2, i32 4, i32 8)
  ret i32 %ret
}

; CHECK-LABEL: Begin function test_i64
; CHECK: %[[#VALUE:]] = OpFunctionParameter %[[#I64]]
; CHECK: %[[#MASK:]] = OpCompositeConstruct %[[#V4I32]] %[[#C1]] %[[#C2]] %[[#C4]] %[[#C8]]
; CHECK: %[[#RET:]] = OpGroupNonUniformIAdd %[[#I64]] %[[#SCOPE]] PartitionedExclusiveScanNV %[[#VALUE]] %[[#MASK]]
; CHECK: OpReturnValue %[[#RET]]
define i64 @test_i64(i64 %value) {
  %ret = call i64 @llvm.spv.subgroup.multi.prefix.sum.i64(
      i64 %value, i32 1, i32 2, i32 4, i32 8)
  ret i64 %ret
}

; CHECK-LABEL: Begin function test_int_vector
; CHECK: %[[#VALUE:]] = OpFunctionParameter %[[#V2I32]]
; CHECK: %[[#MASK:]] = OpCompositeConstruct %[[#V4I32]] %[[#C1]] %[[#C2]] %[[#C4]] %[[#C8]]
; CHECK: %[[#RET:]] = OpGroupNonUniformIAdd %[[#V2I32]] %[[#SCOPE]] PartitionedExclusiveScanNV %[[#VALUE]] %[[#MASK]]
; CHECK: OpReturnValue %[[#RET]]
define <2 x i32> @test_int_vector(<2 x i32> %value) {
  %ret = call <2 x i32> @llvm.spv.subgroup.multi.prefix.sum.v2i32(
      <2 x i32> %value, i32 1, i32 2, i32 4, i32 8)
  ret <2 x i32> %ret
}

; CHECK-LABEL: Begin function test_float_vector
; CHECK: %[[#VALUE:]] = OpFunctionParameter %[[#V4F32]]
; CHECK: %[[#MASK:]] = OpCompositeConstruct %[[#V4I32]] %[[#C1]] %[[#C2]] %[[#C4]] %[[#C8]]
; CHECK: %[[#RET:]] = OpGroupNonUniformFAdd %[[#V4F32]] %[[#SCOPE]] PartitionedExclusiveScanNV %[[#VALUE]] %[[#MASK]]
; CHECK: OpReturnValue %[[#RET]]
define <4 x float> @test_float_vector(<4 x float> %value) {
  %ret = call <4 x float> @llvm.spv.subgroup.multi.prefix.sum.v4f32(
      <4 x float> %value, i32 1, i32 2, i32 4, i32 8)
  ret <4 x float> %ret
}

; CHECK-LABEL: Begin function test_long_vector
; CHECK: %[[#MASK0:]] = OpCompositeConstruct %[[#V4I32]] %[[#C1]] %[[#C2]] %[[#C4]] %[[#C8]]
; CHECK: OpGroupNonUniformFAdd %[[#V4F32]] %[[#SCOPE]] PartitionedExclusiveScanNV %{{[0-9]+}} %[[#MASK0]]
; CHECK: %[[#MASK1:]] = OpCompositeConstruct %[[#V4I32]] %[[#C1]] %[[#C2]] %[[#C4]] %[[#C8]]
; CHECK: OpGroupNonUniformFAdd %[[#F32]] %[[#SCOPE]] PartitionedExclusiveScanNV %{{[0-9]+}} %[[#MASK1]]
define void @test_long_vector() {
  %value = load <5 x float>, ptr addrspace(10) @wide_float
  %ret = call <5 x float> @llvm.spv.subgroup.multi.prefix.sum.v5f32(
      <5 x float> %value, i32 1, i32 2, i32 4, i32 8)
  store <5 x float> %ret, ptr addrspace(10) @wide_float
  ret void
}
