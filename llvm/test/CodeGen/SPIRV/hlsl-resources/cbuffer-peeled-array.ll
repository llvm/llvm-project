; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}


; CHECK-DAG: %[[FLOAT:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[VEC3:[0-9]+]] = OpTypeVector %[[FLOAT]] 3
; CHECK-DAG: %[[I8:[0-9]+]] = OpTypeInt 8 0
; CHECK-DAG: %[[STRUCT_PAD:[0-9]+]] = OpTypeStruct %[[VEC3]] %[[I8]]
; CHECK-DAG: %[[UINT:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[CONST_3:[0-9]+]] = OpConstant %[[UINT]] 3
; CHECK-DAG: %[[ARRAY:[0-9]+]] = OpTypeArray %[[STRUCT_PAD]] %[[CONST_3]]
; CHECK-DAG: %[[CBLAYOUT:[0-9]+]] = OpTypeStruct %[[ARRAY]]
; CHECK-DAG: OpMemberDecorate %[[CBLAYOUT]] 0 Offset 0
; CHECK-DAG: %[[WRAPPER:[0-9]+]] = OpTypeStruct %[[CBLAYOUT]]
; CHECK-DAG: %[[PTR_WRAPPER:[0-9]+]] = OpTypePointer Uniform %[[WRAPPER]]
; CHECK-DAG: %[[ZERO:[0-9]+]] = OpConstant %[[UINT]] 0
; CHECK-DAG: %[[MYCBUFFER:[0-9]+]] = OpVariable %[[PTR_WRAPPER]] Uniform


; TODO(168401): This array stride and offset of element 1 are incorrect. This
; is an issue with how 3 element vectors are handled.
; CHECK-DAG: OpDecorate %[[ARRAY]] ArrayStride 20
; CHECK-DAG: OpMemberDecorate %[[STRUCT_PAD]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[STRUCT_PAD]] 1 Offset 16
; CHECK-DAG: OpMemberDecorate %[[WRAPPER]] 0 Offset 0
; CHECK-DAG: OpDecorate %[[WRAPPER]] Block
%__cblayout_MyCBuffer = type <{ <{ [2 x <{ <3 x float>, target("spirv.Padding", 4) }>], <3 x float> }> }>

@MyCBuffer.cb = local_unnamed_addr global target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) poison
@myArray = external hidden local_unnamed_addr addrspace(12) global <{ [2 x <{ <3 x float>, target("spirv.Padding", 4) }>], <3 x float> }>, align 16
@MyCBuffer.str = private unnamed_addr constant [10 x i8] c"MyCBuffer\00", align 1
@.str = private unnamed_addr constant [7 x i8] c"output\00", align 1

declare target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s___cblayout_MyCBuffers_2_0t(i32, i32, i32, i32, ptr)

define void @main() #1 {
entry:
; CHECK: %[[BUFFER_HANDLE:[0-9]+]] = OpCopyObject %[[PTR_WRAPPER]] %[[MYCBUFFER]]
; CHECK: %[[ACCESS_ARRAY:[0-9]+]] = OpAccessChain {{%[0-9]+}} %[[BUFFER_HANDLE]] %[[ZERO]] %[[ZERO]]
  %MyCBuffer.cb_h.i.i = tail call target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_s___cblayout_MyCBuffers_2_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @MyCBuffer.str)
  store target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) %MyCBuffer.cb_h.i.i, ptr @MyCBuffer.cb, align 8

  %0 = tail call target("spirv.VulkanBuffer", [0 x <3 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v3f32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %1 = tail call i32 @llvm.spv.thread.id.i32(i32 0)

; CHECK: %[[IDX:[0-9]+]] = OpUMod %[[UINT]] {{%[0-9]+}} %[[CONST_3]]
  %rem.i = urem i32 %1, 3

; CHECK: %[[IDX_CONV:[0-9]+]] = OpUConvert {{.*}} %[[IDX]]
  %idxprom.i = zext nneg i32 %rem.i to i64

; CHECK: %[[PTR_ELEM:[0-9]+]] = OpAccessChain {{%[0-9]+}} %[[ACCESS_ARRAY]] %[[IDX_CONV]]
  %cbufferidx.i = getelementptr <{ <3 x float>, target("spirv.Padding", 4) }>, ptr addrspace(12) @myArray, i64 %idxprom.i

; CHECK: %[[PTR_FIELD:[0-9]+]] = OpAccessChain {{%[0-9]+}} %[[PTR_ELEM]] {{.*}}
; CHECK: %[[VAL_VEC3:[0-9]+]] = OpLoad %[[VEC3]] %[[PTR_FIELD]] Aligned 16
  %2 = load <3 x float>, ptr addrspace(12) %cbufferidx.i, align 16

  %3 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v3f32_12_1t(target("spirv.VulkanBuffer", [0 x <3 x float>], 12, 1) %0, i32 %1)
  store <3 x float> %2, ptr addrspace(11) %3, align 16
  ret void
}

declare i32 @llvm.spv.thread.id.i32(i32)

declare target("spirv.VulkanBuffer", [0 x <3 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v3f32_12_1t(i32, i32, i32, i32, ptr)

declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v3f32_12_1t(target("spirv.VulkanBuffer", [0 x <3 x float>], 12, 1), i32)

attributes #1 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }

!hlsl.cbs = !{!0}

!0 = !{ptr @MyCBuffer.cb, ptr addrspace(12) @myArray}
