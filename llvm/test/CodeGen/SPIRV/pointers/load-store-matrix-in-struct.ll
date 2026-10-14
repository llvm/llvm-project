; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -stop-after=spirv-legalize-pointer-cast -o - | FileCheck %s --check-prefix=IRCHECK
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3 %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: [[FLOAT:%[0-9]+]] = OpTypeFloat 32
; CHECK-DAG: [[VEC2FLOAT:%[0-9]+]] = OpTypeVector [[FLOAT]] 2
; CHECK-DAG: [[UINT:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[UINT4:%[0-9]+]] = OpConstant [[UINT]] 4
; CHECK-DAG: [[ARRAY4VEC2:%[0-9]+]] = OpTypeArray [[VEC2FLOAT]] [[UINT4]]
; CHECK-DAG: [[STRUCT:%[0-9]+]] = OpTypeStruct [[ARRAY4VEC2]]

; Load:
; CHECK: [[IN_AC_STRUCT:%[0-9]+]] = OpAccessChain %{{[0-9]+}} %{{[0-9]+}} %{{[0-9]+}} %{{[0-9]+}}
; CHECK: [[IN_AC_ARR:%[0-9]+]] = OpAccessChain %{{[0-9]+}} [[IN_AC_STRUCT]] %{{[0-9]+}}
; CHECK: [[IN_AC_VEC:%[0-9]+]] = OpAccessChain %{{[0-9]+}} [[IN_AC_ARR]] %{{[0-9]+}}
; CHECK: OpLoad [[VEC2FLOAT]] [[IN_AC_VEC]]

; Store:
; CHECK: [[OUT_AC_STRUCT:%[0-9]+]] = OpAccessChain %{{[0-9]+}} %{{[0-9]+}} %{{[0-9]+}} %{{[0-9]+}}
; CHECK: [[OUT_AC_ARR:%[0-9]+]] = OpInBoundsAccessChain %{{[0-9]+}} [[OUT_AC_STRUCT]] %{{[0-9]+}}
; CHECK: [[OUT_AC_VEC:%[0-9]+]] = OpAccessChain %{{[0-9]+}} [[OUT_AC_ARR]] %{{[0-9]+}}
; CHECK: OpStore [[OUT_AC_VEC]]

; IRCHECK-LABEL: define void @main

; IRCHECK: [[IN_PTR:%[0-9]+]] = {{.*}}call {{.*}} ptr addrspace(11) @llvm.spv.resource.getpointer
; IRCHECK: [[GEP_STRUCT_IN:%[0-9]+]] = call ptr addrspace(11) (i1, ptr addrspace(11), ...) @llvm.spv.gep.p11.p11(i1 false, ptr addrspace(11) [[IN_PTR]], i32 0, i32 0)
; IRCHECK: [[GEP_ROW_IN:%[0-9]+]] = call ptr addrspace(11) (i1, ptr addrspace(11), ...) @llvm.spv.gep.p11.p11(i1 false, ptr addrspace(11) [[GEP_STRUCT_IN]], i32 0, i32 0)
; IRCHECK: load <2 x float>, ptr addrspace(11) [[GEP_ROW_IN]]

; IRCHECK: [[OUT_PTR:%[0-9]+]] = {{.*}}call {{.*}} ptr addrspace(11) @llvm.spv.resource.getpointer
; IRCHECK: [[GEP_STRUCT_OUT:%[0-9]+]] = call ptr addrspace(11) (i1, ptr addrspace(11), ...) @llvm.spv.gep.p11.p11(i1 true, ptr addrspace(11) [[OUT_PTR]], i32 0, i32 0)
; IRCHECK: [[GEP_ROW_OUT:%[0-9]+]] = call ptr addrspace(11) (i1, ptr addrspace(11), ...) @llvm.spv.gep.p11.p11(i1 false, ptr addrspace(11) [[GEP_STRUCT_OUT]], i32 0, i32 0)
; IRCHECK: store <2 x float> {{%[0-9]+}}, ptr addrspace(11) [[GEP_ROW_OUT]]

%struct.S = type { [4 x <2 x float>] }

@.str = private unnamed_addr constant [3 x i8] c"In\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"Out\00", align 1

define void @main() #0 {
entry:
  %0 = tail call target("spirv.VulkanBuffer", [0 x %struct.S], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str)
  %1 = tail call noundef align 4 dereferenceable(32) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 0) %0, i32 0)
  %2 = load <8 x float>, ptr addrspace(11) %1, align 4

  %3 = tail call target("spirv.VulkanBuffer", [0 x %struct.S], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.str.2)
  %4 = tail call noundef align 4 dereferenceable(32) ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 0) %3, i32 0)
  store <8 x float> %2, ptr addrspace(11) %4, align 4

  ret void
}

declare target("spirv.VulkanBuffer", [0 x %struct.S], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer(target("spirv.VulkanBuffer", [0 x %struct.S], 12, 0), i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
