; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[FLOAT:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[VEC4:[0-9]+]] = OpTypeVector %[[FLOAT]] 4
; CHECK-DAG: %[[PTR_VEC4:[0-9]+]] = OpTypePointer Uniform %[[VEC4]]
; CHECK-DAG: %[[INT:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[PTR_INT:[0-9]+]] = OpTypePointer Uniform %[[INT]]
; CHECK-DAG: %[[INT64:[0-9]+]] = OpTypeInt 64 0
; CHECK-DAG: %[[CONST_4:[0-9]+]] = OpConstant %[[INT]] 4{{$}}

; CHECK-DAG: %[[ARRAY:[0-9]+]] = OpTypeArray %[[VEC4]] %[[CONST_4]]
; CHECK-DAG: %[[PTR_ARRAY:[0-9]+]] = OpTypePointer Uniform %[[ARRAY]]

; CHECK-DAG: %[[STRUCT_INNER:[0-9]+]] = OpTypeStruct %[[ARRAY]] %[[INT]]
; CHECK-DAG: %[[STRUCT_CBUFFER:[0-9]+]] = OpTypeStruct %[[STRUCT_INNER]]
; CHECK-DAG: %[[PTR_CBUFFER:[0-9]+]] = OpTypePointer Uniform %[[STRUCT_CBUFFER]]

; CHECK-DAG: OpDecorate %[[ARRAY]] ArrayStride 16
; CHECK-DAG: OpMemberDecorate %[[STRUCT_INNER]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[STRUCT_INNER]] 1 Offset 64
; CHECK-DAG: OpMemberDecorate %[[STRUCT_CBUFFER]] 0 Offset 0
; CHECK-DAG: OpDecorate %[[STRUCT_CBUFFER]] Block

; CHECK-DAG: %[[ZERO:[0-9]+]] = OpConstant %[[INT]] 0{{$}}
; CHECK-DAG: %[[ONE:[0-9]+]] = OpConstant %[[INT]] 1{{$}}

; CHECK: %[[CBUFFER:[0-9]+]] = OpVariable %[[PTR_CBUFFER]] Uniform

%__cblayout_MyCBuffer = type <{ [4 x <4 x float>], i32 }>

@MyCBuffer.cb = local_unnamed_addr global target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) poison
@colors = external hidden local_unnamed_addr addrspace(12) global [4 x <4 x float>], align 16
@index = external hidden local_unnamed_addr addrspace(12) global i32, align 4
@MyCBuffer.str = private unnamed_addr constant [10 x i8] c"MyCBuffer\00", align 1
@.str = private unnamed_addr constant [7 x i8] c"output\00", align 1

declare target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s___cblayout_MyCBuffers_2_0t(i32, i32, i32, i32, ptr)

define void @main() #1 {
entry:
; Get pointers to the two elements of the cbuffer
; CHECK: %[[COPY:[0-9]+]] = OpCopyObject %[[PTR_CBUFFER]] %[[CBUFFER]]
; CHECK: %[[PTR_ARRAY_ACCESS:[0-9]+]] = OpAccessChain %[[PTR_ARRAY]] %[[COPY]] %[[ZERO]] %[[ZERO]]
; CHECK: %[[PTR_INT_ACCESS:[0-9]+]] = OpAccessChain %[[PTR_INT]] %[[COPY]] %[[ZERO]] %[[ONE]]
  %MyCBuffer.cb_h.i.i = tail call target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s___cblayout_MyCBuffers_2_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @MyCBuffer.str)
  store target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) %MyCBuffer.cb_h.i.i, ptr @MyCBuffer.cb, align 8

  %0 = tail call target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)

; CHECK: %[[VAL_INT:[0-9]+]] = OpLoad %[[INT]] %[[PTR_INT_ACCESS]] Aligned 4
  %1 = load i32, ptr addrspace(12) @index, align 4

; CHECK: %[[VAL_INT64:[0-9]+]] = OpSConvert %[[INT64]] %[[VAL_INT]]
  %idxprom.i = sext i32 %1 to i64

; CHECK: %[[PTR_ELEM:[0-9]+]] = OpInBoundsAccessChain %[[PTR_VEC4]] %[[PTR_ARRAY_ACCESS]] %[[VAL_INT64]]
  %arrayidx.i = getelementptr inbounds <4 x float>, ptr addrspace(12) @colors, i64 %idxprom.i

; CHECK: %[[VAL_ELEM:[0-9]+]] = OpLoad %[[VEC4]] %[[PTR_ELEM]] Aligned 16
  %2 = load <4 x float>, ptr addrspace(12) %arrayidx.i, align 16

; CHECK: OpStore {{%[0-9]+}} %[[VAL_ELEM]] Aligned 16
  %3 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) %0, i32 0)
  store <4 x float> %2, ptr addrspace(11) %3, align 16
  ret void
}

declare target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_1t(i32, i32, i32, i32, ptr)

declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1), i32)

attributes #1 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!hlsl.cbs = !{!0}

!0 = !{ptr @MyCBuffer.cb, ptr addrspace(12) @colors, ptr addrspace(12) @index}
