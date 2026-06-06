; A GLCompute kernel using SPV_KHR_ray_query against a descriptor-bound
; acceleration structure (a UniformConstant global decorated DescriptorSet and
; Binding, materialized via llvm.spv.resource.handlefrombinding). It initializes
; a ray, proceeds over candidates, inspects the candidate intersection type, and
; reads the committed type. The module must pass spirv-val --target-env vulkan1.3.

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_ray_query %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_ray_query %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: OpCapability RayQueryKHR
; CHECK-DAG: OpExtension "SPV_KHR_ray_query"
; CHECK-DAG: OpEntryPoint GLCompute %[[#entry:]] "main"
; CHECK-DAG: OpDecorate %[[#AS:]] DescriptorSet 0
; CHECK-DAG: OpDecorate %[[#AS]] Binding 0
; CHECK-DAG: %[[#ASTy:]] = OpTypeAccelerationStructureKHR
; CHECK-DAG: %[[#RQTy:]] = OpTypeRayQueryKHR
; CHECK: %[[#RQ:]] = OpVariable %[[#]] Function
; CHECK: OpRayQueryInitializeKHR %[[#RQ]]
; CHECK: %[[#]] = OpRayQueryProceedKHR
; CHECK: %[[#]] = OpRayQueryGetIntersectionTypeKHR

@.str.as = private unnamed_addr constant [3 x i8] c"as\00", align 1

define void @main() local_unnamed_addr #0 {
entry:
  %as = tail call target("spirv.AccelerationStructureKHR")
      @llvm.spv.resource.handlefrombinding.tspirv.AccelerationStructureKHR(
          i32 0, i32 0, i32 1, i32 0, ptr nonnull @.str.as)
  %rq = alloca target("spirv.RayQueryKHR"), align 8
  call void @llvm.spv.ray.query.initialize(ptr %rq,
      target("spirv.AccelerationStructureKHR") %as,
      i32 0, i32 255,
      <3 x float> zeroinitializer, float 0.000000e+00,
      <3 x float> <float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>,
      float 0.000000e+00)
  br label %loop.header

loop.header:
  %more = call i1 @llvm.spv.ray.query.proceed(ptr %rq)
  br i1 %more, label %loop.body, label %loop.exit

loop.body:
  %cand = call i32 @llvm.spv.ray.query.get.intersection.type(ptr %rq, i32 0)
  br label %loop.header

loop.exit:
  %committed = call i32 @llvm.spv.ray.query.get.intersection.type(ptr %rq, i32 1)
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
