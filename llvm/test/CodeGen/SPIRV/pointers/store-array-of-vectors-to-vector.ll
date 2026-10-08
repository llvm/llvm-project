; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -stop-after=spirv-legalize-pointer-cast -o - | FileCheck %s --check-prefix=IRCHECK
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 --scalar-block-layout %}

; CHECK-DAG: [[FLOAT:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[VEC3FLOAT:%[0-9]+]] = OpTypeVector [[FLOAT]] 3
; CHECK-DAG: [[PTR_VEC3:%[0-9]+]] = OpTypePointer StorageBuffer [[VEC3FLOAT]]
; CHECK-DAG: [[UINT:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[UINT2:%[0-9]+]] = OpConstant [[UINT]] 2
; CHECK-DAG: [[ARRAY2VEC3:%[0-9]+]] = OpTypeArray [[VEC3FLOAT]] [[UINT2]]
; CHECK-DAG: [[PTR_ARRAY2VEC3:%[0-9]+]] = OpTypePointer StorageBuffer [[ARRAY2VEC3]]
; CHECK-DAG: [[UINT0:%[0-9]+]] = OpConstant [[UINT]] 0
; CHECK-DAG: [[UINT1:%[0-9]+]] = OpConstant [[UINT]] 1
; CHECK-DAG: [[UNDEF_VEC3:%[0-9]+]] = OpUndef [[VEC3FLOAT]]

; Load from input[0][0] (first vector)
; CHECK:      [[IN_AC_ARR:%[0-9]+]] = OpAccessChain [[PTR_ARRAY2VEC3]] %{{[0-9]+}} [[UINT0]] [[UINT0]]
; CHECK-NEXT: [[IN_AC_VEC0:%[0-9]+]] = OpAccessChain [[PTR_VEC3]] [[IN_AC_ARR]] [[UINT0]]
; CHECK-NEXT: [[LOAD0:%[0-9]+]] = OpLoad [[VEC3FLOAT]] [[IN_AC_VEC0]]
; CHECK-NEXT: [[EX_0_0:%[0-9]+]] = OpCompositeExtract [[FLOAT]] [[LOAD0]] 0
; CHECK-NEXT: [[IN_AC_VEC0_2:%[0-9]+]] = OpAccessChain [[PTR_VEC3]] [[IN_AC_ARR]] [[UINT0]]
; CHECK-NEXT: [[LOAD1:%[0-9]+]] = OpLoad [[VEC3FLOAT]] [[IN_AC_VEC0_2]]
; CHECK-NEXT: [[EX_0_1:%[0-9]+]] = OpCompositeExtract [[FLOAT]] [[LOAD1]] 1
; CHECK-NEXT: [[IN_AC_VEC0_3:%[0-9]+]] = OpAccessChain [[PTR_VEC3]] [[IN_AC_ARR]] [[UINT0]]
; CHECK-NEXT: [[LOAD2:%[0-9]+]] = OpLoad [[VEC3FLOAT]] [[IN_AC_VEC0_3]]
; CHECK-NEXT: [[EX_0_2:%[0-9]+]] = OpCompositeExtract [[FLOAT]] [[LOAD2]] 2

; Load from input[0][1] (second vector)
; CHECK-NEXT: [[IN_AC_VEC1:%[0-9]+]] = OpAccessChain [[PTR_VEC3]] [[IN_AC_ARR]] [[UINT1]]
; CHECK-NEXT: [[LOAD3:%[0-9]+]] = OpLoad [[VEC3FLOAT]] [[IN_AC_VEC1]]
; CHECK-NEXT: [[EX_1_0:%[0-9]+]] = OpCompositeExtract [[FLOAT]] [[LOAD3]] 0
; CHECK-NEXT: [[IN_AC_VEC1_2:%[0-9]+]] = OpAccessChain [[PTR_VEC3]] [[IN_AC_ARR]] [[UINT1]]
; CHECK-NEXT: [[LOAD4:%[0-9]+]] = OpLoad [[VEC3FLOAT]] [[IN_AC_VEC1_2]]
; CHECK-NEXT: [[EX_1_1:%[0-9]+]] = OpCompositeExtract [[FLOAT]] [[LOAD4]] 1
; CHECK-NEXT: [[IN_AC_VEC1_3:%[0-9]+]] = OpAccessChain [[PTR_VEC3]] [[IN_AC_ARR]] [[UINT1]]
; CHECK-NEXT: [[LOAD5:%[0-9]+]] = OpLoad [[VEC3FLOAT]] [[IN_AC_VEC1_3]]
; CHECK-NEXT: [[EX_1_2:%[0-9]+]] = OpCompositeExtract [[FLOAT]] [[LOAD5]] 2

; Store to output[0][0] (first vector)
; CHECK-NEXT: [[OUT_AC_ARR:%[0-9]+]] = OpAccessChain [[PTR_ARRAY2VEC3]] %{{[0-9]+}} [[UINT0]] [[UINT0]]
; CHECK-NEXT: [[OUT_AC_VEC0:%[0-9]+]] = OpAccessChain [[PTR_VEC3]] [[OUT_AC_ARR]] [[UINT0]]
; CHECK-NEXT: [[INS_0_0:%[0-9]+]] = OpCompositeInsert [[VEC3FLOAT]] [[EX_0_0]] [[UNDEF_VEC3]] 0
; CHECK-NEXT: [[INS_0_1:%[0-9]+]] = OpCompositeInsert [[VEC3FLOAT]] [[EX_0_1]] [[INS_0_0]] 1
; CHECK-NEXT: [[INS_0_2:%[0-9]+]] = OpCompositeInsert [[VEC3FLOAT]] [[EX_0_2]] [[INS_0_1]] 2
; CHECK-NEXT: OpStore [[OUT_AC_VEC0]] [[INS_0_2]]

; Store to output[0][1] (second vector)
; CHECK-NEXT: [[OUT_AC_VEC1:%[0-9]+]] = OpAccessChain [[PTR_VEC3]] [[OUT_AC_ARR]] [[UINT1]]
; CHECK-NEXT: [[INS_1_0:%[0-9]+]] = OpCompositeInsert [[VEC3FLOAT]] [[EX_1_0]] [[UNDEF_VEC3]] 0
; CHECK-NEXT: [[INS_1_1:%[0-9]+]] = OpCompositeInsert [[VEC3FLOAT]] [[EX_1_1]] [[INS_1_0]] 1
; CHECK-NEXT: [[INS_1_2:%[0-9]+]] = OpCompositeInsert [[VEC3FLOAT]] [[EX_1_2]] [[INS_1_1]] 2
; CHECK-NEXT: OpStore [[OUT_AC_VEC1]] [[INS_1_2]]
; CHECK-NEXT: OpReturn

; IRCHECK-LABEL: define void @main

; Load: GEP to input row 0, load <3 x float>, extract elements
; IRCHECK: [[IN_PTR:%[0-9]+]] = {{.*}}call {{.*}} ptr addrspace(11) @llvm.spv.resource.getpointer
; IRCHECK: [[GEP_ROW0:%[0-9]+]] = call ptr addrspace(11) (i1, ptr addrspace(11), ...) @llvm.spv.gep.p11.p11(i1 false, ptr addrspace(11) [[IN_PTR]], i32 0, i32 0)
; IRCHECK: [[VEC_ROW0:%[0-9]+]] = load <3 x float>, ptr addrspace(11) [[GEP_ROW0]]
; IRCHECK: [[E00:%[0-9]+]] = call float @llvm.spv.extractelt.f32.v3f32.i32(<3 x float> [[VEC_ROW0]], i32 0)
; IRCHECK: [[E01:%[0-9]+]] = call float @llvm.spv.extractelt.f32.v3f32.i32(<3 x float> {{%[0-9]+}}, i32 1)
; IRCHECK: [[E02:%[0-9]+]] = call float @llvm.spv.extractelt.f32.v3f32.i32(<3 x float> {{%[0-9]+}}, i32 2)

; Load: GEP to input row 1, load <3 x float>, extract elements
; IRCHECK: [[GEP_ROW1:%[0-9]+]] = call ptr addrspace(11) (i1, ptr addrspace(11), ...) @llvm.spv.gep.p11.p11(i1 false, ptr addrspace(11) [[IN_PTR]], i32 0, i32 1)
; IRCHECK: [[VEC_ROW1:%[0-9]+]] = load <3 x float>, ptr addrspace(11) [[GEP_ROW1]]
; IRCHECK: [[E10:%[0-9]+]] = call float @llvm.spv.extractelt.f32.v3f32.i32(<3 x float> [[VEC_ROW1]], i32 0)
; IRCHECK: [[E11:%[0-9]+]] = call float @llvm.spv.extractelt.f32.v3f32.i32(<3 x float> {{%[0-9]+}}, i32 1)
; IRCHECK: [[E12:%[0-9]+]] = call float @llvm.spv.extractelt.f32.v3f32.i32(<3 x float> {{%[0-9]+}}, i32 2)

; Store: GEP to output row 0, extract from <6 x float>, insert into <3 x float>, store
; IRCHECK: [[OUT_PTR:%[0-9]+]] = {{.*}}call {{.*}} ptr addrspace(11) @llvm.spv.resource.getpointer
; IRCHECK: [[OUT_GEP_ROW0:%[0-9]+]] = call ptr addrspace(11) (i1, ptr addrspace(11), ...) @llvm.spv.gep.p11.p11(i1 false, ptr addrspace(11) [[OUT_PTR]], i32 0, i32 0)
; IRCHECK: call float @llvm.spv.extractelt.f32.v6f32.i32(<6 x float> {{%[0-9]+}}, i32 0)
; IRCHECK: call float @llvm.spv.extractelt.f32.v6f32.i32(<6 x float> {{%[0-9]+}}, i32 1)
; IRCHECK: call float @llvm.spv.extractelt.f32.v6f32.i32(<6 x float> {{%[0-9]+}}, i32 2)
; IRCHECK: call <3 x float> @llvm.spv.insertelt.v3f32.v3f32.f32.i32(<3 x float> poison, float {{%[0-9]+}}, i32 0)
; IRCHECK: call <3 x float> @llvm.spv.insertelt.v3f32.v3f32.f32.i32(<3 x float> {{%[0-9]+}}, float {{%[0-9]+}}, i32 1)
; IRCHECK: [[STORE_VEC0:%[0-9]+]] = call <3 x float> @llvm.spv.insertelt.v3f32.v3f32.f32.i32(<3 x float> {{%[0-9]+}}, float {{%[0-9]+}}, i32 2)
; IRCHECK: store <3 x float> [[STORE_VEC0]], ptr addrspace(11) [[OUT_GEP_ROW0]]

; Store: GEP to output row 1, extract from <6 x float>, insert into <3 x float>, store
; IRCHECK: [[OUT_GEP_ROW1:%[0-9]+]] = call ptr addrspace(11) (i1, ptr addrspace(11), ...) @llvm.spv.gep.p11.p11(i1 false, ptr addrspace(11) [[OUT_PTR]], i32 0, i32 1)
; IRCHECK: call float @llvm.spv.extractelt.f32.v6f32.i32(<6 x float> {{%[0-9]+}}, i32 3)
; IRCHECK: call float @llvm.spv.extractelt.f32.v6f32.i32(<6 x float> {{%[0-9]+}}, i32 4)
; IRCHECK: call float @llvm.spv.extractelt.f32.v6f32.i32(<6 x float> {{%[0-9]+}}, i32 5)
; IRCHECK: call <3 x float> @llvm.spv.insertelt.v3f32.v3f32.f32.i32(<3 x float> poison, float {{%[0-9]+}}, i32 0)
; IRCHECK: call <3 x float> @llvm.spv.insertelt.v3f32.v3f32.f32.i32(<3 x float> {{%[0-9]+}}, float {{%[0-9]+}}, i32 1)
; IRCHECK: [[STORE_VEC1:%[0-9]+]] = call <3 x float> @llvm.spv.insertelt.v3f32.v3f32.f32.i32(<3 x float> {{%[0-9]+}}, float {{%[0-9]+}}, i32 2)
; IRCHECK: store <3 x float> [[STORE_VEC1]], ptr addrspace(11) [[OUT_GEP_ROW1]]

@.str = private unnamed_addr constant [4 x i8] c"InA\00", align 1
@.str.2 = private unnamed_addr constant [5 x i8] c"OutA\00", align 1

define void @main() local_unnamed_addr #0 {
entry:
  %0 = tail call target("spirv.VulkanBuffer", [0 x [2 x <3 x float>]], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %1 = tail call target("spirv.VulkanBuffer", [0 x [2 x <3 x float>]], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer(i32 0, i32 2, i32 1, i32 0, ptr nonnull @.str.2)
  %2 = tail call noundef align 4 dereferenceable(24) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer(target("spirv.VulkanBuffer", [0 x [2 x <3 x float>]], 12, 0) %0, i32 0)
  %3 = load <6 x float>, ptr addrspace(11) %2, align 4
  %4 = tail call noundef align 4 dereferenceable(24) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer(target("spirv.VulkanBuffer", [0 x [2 x <3 x float>]], 12, 1) %1, i32 0)
  store <6 x float> %3, ptr addrspace(11) %4, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
