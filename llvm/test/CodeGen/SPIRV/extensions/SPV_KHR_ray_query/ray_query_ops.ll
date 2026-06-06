; SPV_KHR_ray_query operations via llvm.spv.ray.query.* intrinsics and GlobalISel
; selection, used because the __spirv_* builtin path is OpenCL-only and cannot
; lower ray query, which is a Vulkan-only extension.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_ray_query %s -o - | FileCheck %s

; CHECK-DAG: OpCapability RayQueryKHR
; CHECK-DAG: OpExtension "SPV_KHR_ray_query"
; CHECK-DAG: %[[#RQTy:]] = OpTypeRayQueryKHR
; CHECK: OpRayQueryInitializeKHR
; CHECK: %[[#]] = OpRayQueryProceedKHR
; CHECK: %[[#]] = OpRayQueryGetIntersectionTypeKHR

define void @ray_query_ops(target("spirv.AccelerationStructureKHR") %as) {
entry:
  %rq = alloca target("spirv.RayQueryKHR"), align 8
  call void @llvm.spv.ray.query.initialize(ptr %rq, target("spirv.AccelerationStructureKHR") %as, i32 0, i32 255, <3 x float> zeroinitializer, float 0.000000e+00, <3 x float> <float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, float 1.000000e+03)
  %p = call i1 @llvm.spv.ray.query.proceed(ptr %rq)
  %t = call i32 @llvm.spv.ray.query.get.intersection.type(ptr %rq, i32 0)
  ret void
}
