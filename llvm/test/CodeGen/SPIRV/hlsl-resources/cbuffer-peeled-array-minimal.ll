; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpDecorate %[[ARRAY:[0-9]+]] ArrayStride 16
; CHECK-DAG: OpMemberDecorate %[[CBLAYOUT:[0-9]+]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[CBLAYOUT]] 1 Offset 52
; CHECK-DAG: OpMemberDecorate %[[WRAPPER:[0-9]+]] 0 Offset 0
; CHECK-DAG: OpDecorate %[[WRAPPER]] Block
; CHECK-DAG: OpMemberDecorate %[[STRUCT:[0-9]+]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[STRUCT_PAD:[0-9]+]] 0 Offset 0
; CHECK-DAG: OpMemberDecorate %[[STRUCT_PAD]] 1 Offset 4

; CHECK-DAG: %[[FLOAT:[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: %[[STRUCT]] = OpTypeStruct %[[FLOAT]]
; CHECK-DAG: %[[I8:[0-9]+]] = OpTypeInt 8 0
; CHECK-DAG: %[[STRUCT_PAD]] = OpTypeStruct %[[STRUCT]] %[[I8]]
; CHECK-DAG: %[[UINT:[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: %[[CONST_4:[0-9]+]] = OpConstant %[[UINT]] 4
; CHECK-DAG: %[[ARRAY]] = OpTypeArray %[[STRUCT_PAD]] %[[CONST_4]]
; CHECK-DAG: %[[CBLAYOUT]] = OpTypeStruct %[[ARRAY]] %[[FLOAT]]
; CHECK-DAG: %[[WRAPPER]] = OpTypeStruct %[[CBLAYOUT]]
; CHECK-DAG: %[[PTR_WRAPPER:[0-9]+]] = OpTypePointer Uniform %[[WRAPPER]]
; CHECK-DAG: %[[ZERO:[0-9]+]] = OpConstant %[[UINT]] 0
; CHECK-DAG: %[[MYCBUFFER:[0-9]+]] = OpVariable %[[PTR_WRAPPER]] Uniform

; CHECK-DAG: %[[I64:[0-9]+]] = OpTypeInt 64 0
; CHECK-DAG: %[[STRUCT2:[0-9]+]] = OpTypeStruct %[[I64]] %[[UINT]]
; CHECK-DAG: %[[CONST_3:[0-9]+]] = OpConstant %[[UINT]] 3
; CHECK-DAG: %[[ARRAY2:[0-9]+]] = OpTypeArray %[[STRUCT2]] %[[CONST_3]]
; CHECK-DAG: %[[CBLAYOUT2:[0-9]+]] = OpTypeStruct %[[ARRAY2]] %[[I64]]
; CHECK-DAG: %[[PTR_PRIVATE:[0-9]+]] = OpTypePointer Private %[[CBLAYOUT2]]
; CHECK-DAG: %[[MYPRIVATEVAR:[0-9]+]] = OpVariable %[[PTR_PRIVATE]] Private

%__cblayout_MyCBuffer = type <{ <{ [3 x <{ %OrigType, target("spirv.Padding", 12) }>], %OrigType }>, float }>
%OrigType = type <{ float }>

%__cblayout_MyCBuffer2 = type <{ [ 3 x <{ i64, i32 }> ], i64 }>

@MyCBuffer.cb = local_unnamed_addr global target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) poison
@myPrivateVar = internal addrspace(10) global %__cblayout_MyCBuffer2 poison

@myArray = external hidden local_unnamed_addr addrspace(12) global <{ [3 x <{ %OrigType, target("spirv.Padding", 12) }>], %OrigType }>, align 1
@MyCBuffer.str = private unnamed_addr constant [10 x i8] c"MyCBuffer\00", align 1
@.str = private unnamed_addr constant [7 x i8] c"output\00", align 1

declare target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_s___cblayout_MyCBuffers_2_0t(i32, i32, i32, i32, ptr)

define void @main() #1 {
entry:
; CHECK: %[[BUFFER_HANDLE:[0-9]+]] = OpCopyObject %[[PTR_WRAPPER]] %[[MYCBUFFER]]
; CHECK: %[[ACCESS_ARRAY:[0-9]+]] = OpAccessChain {{%[0-9]+}} %[[BUFFER_HANDLE]] %[[ZERO]] %[[ZERO]]
  %MyCBuffer.cb_h.i.i = tail call target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) @llvm.spv.resource.handlefromimplicitbinding.tspirv.VulkanBuffer_s___cblayout_MyCBuffers_2_0t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @MyCBuffer.str)
  store target("spirv.VulkanBuffer", %__cblayout_MyCBuffer, 2, 0) %MyCBuffer.cb_h.i.i, ptr @MyCBuffer.cb, align 8

  %0 = tail call target("spirv.Image", float, 5, 2, 0, 0, 2, 1) @llvm.spv.resource.handlefromimplicitbinding.tspirv.Image_f32_5_2_0_0_2_1t(i32 1, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %1 = tail call i32 @llvm.spv.thread.id.i32(i32 0)
  %rem.i = and i32 %1, 3

; CHECK: %[[IDX_CONV:[0-9]+]] = OpUConvert {{.*}}
  %idxprom.i = zext nneg i32 %rem.i to i64

; CHECK: %[[PTR_ELEM:[0-9]+]] = OpAccessChain {{%[0-9]+}} %[[ACCESS_ARRAY]] %[[IDX_CONV]]
  %cbufferidx.i = getelementptr <{ %OrigType, target("spirv.Padding", 12) }>, ptr addrspace(12) @myArray, i64 %idxprom.i

; CHECK: %[[PTR_FIELD:[0-9]+]] = OpAccessChain {{%[0-9]+}} %[[PTR_ELEM]] %[[ZERO]] %[[ZERO]]
; CHECK: %[[VAL_FLOAT:[0-9]+]] = OpLoad %[[FLOAT]] %[[PTR_FIELD]] Aligned 4
  %2 = load float, ptr addrspace(12) %cbufferidx.i, align 4

  %val = load i64, ptr addrspace(10) getelementptr (%__cblayout_MyCBuffer2, ptr addrspace(10) @myPrivateVar, i32 0, i32 1), align 8
  %val.float = sitofp i64 %val to float

  %vecinit4.i = insertelement <4 x float> <float poison, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, float %2, i64 0
  %vecinit4.i.2 = insertelement <4 x float> %vecinit4.i, float %val.float, i64 1
  %3 = tail call noundef align 16 dereferenceable(16) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_1t(target("spirv.Image", float, 5, 2, 0, 0, 2, 1) %0, i32 0)
  store <4 x float> %vecinit4.i.2, ptr addrspace(11) %3, align 16
; CHECK: OpImageWrite {{%[0-9]+}} {{%[0-9]+}} {{%[0-9]+}}
  ret void
}

declare i32 @llvm.spv.thread.id.i32(i32)

declare target("spirv.Image", float, 5, 2, 0, 0, 2, 1) @llvm.spv.resource.handlefromimplicitbinding.tspirv.Image_f32_5_2_0_0_2_1t(i32, i32, i32, i32, ptr)

declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.Image_f32_5_2_0_0_2_1t(target("spirv.Image", float, 5, 2, 0, 0, 2, 1), i32)

attributes #1 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }

!hlsl.cbs = !{!0}

!0 = distinct !{ptr @MyCBuffer.cb, ptr addrspace(12) @myArray, null}
