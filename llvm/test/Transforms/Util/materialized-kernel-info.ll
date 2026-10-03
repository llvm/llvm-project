; RUN: opt -S -passes='materialize-kernel-info' < %s | FileCheck %s

; This test checks if kernel info is correctly materialized for `ptx_kernel`,
; `amdgpu_kernel` and `spir_kernel` kernels.

; CHECK-DAG: @cuda_kernel_kernel_info = constant [4 x i8] c"\00 \00\00"
define ptx_kernel void @cuda_kernel(i32 %x) {
  ret void
}

; CHECK-DAG: @hip_kernel_kernel_info = constant [4 x i8] c"\00@\00\00"
define amdgpu_kernel void @hip_kernel(i64 %x) {
  ret void
}

; CHECK-DAG: @__omp_offloading_nvptx_l42_kernel_info = constant [4 x i8] c"\03\00\00\00"
define ptx_kernel void @__omp_offloading_nvptx_l42(ptr %dyn) "kernel" {
  ret void
}

; CHECK-DAG: @__omp_offloading_amdgpu_l42_kernel_info = constant [4 x i8] c"\03\00\00\00"
define amdgpu_kernel void @__omp_offloading_amdgpu_l42(ptr %dyn) "kernel" {
  ret void
}

; CHECK-DAG: @spir_kernel_kernel_info = constant [4 x i8] c"\01\00\00\00"
define spir_kernel void @spir_kernel(float %x) {
  ret void
}

; CHECK-DAG: @typed_kernel_kernel_info = constant [20 x i8] c"\00 \00\00\00@\00\00\01\00\00\00\02\00\00\00\03\00\00\00"
define amdgpu_kernel void @typed_kernel(i32 %i32, i64 %i64, float %f32,
                                        double %f64, ptr %p) {
  ret void
}

; CHECK-DAG: @vector_unknown_kernel_kernel_info = constant [4 x i8] c"\FF\00\00\00"
define amdgpu_kernel void @vector_unknown_kernel(<2 x i32> %v) {
  ret void
}
