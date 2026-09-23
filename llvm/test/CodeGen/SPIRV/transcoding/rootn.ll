;; Check that backend converts scalar arg to vector for rootn math instructions.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable

;; __kernel void test_kernel_float(float3 x, int n, __global float3* ret) {
;;    *ret = rootn(x, n);
;; }

; CHECK-SPIRV: %{{.*}} rootn

define dso_local spir_kernel void @test_kernel_float(<3 x float> noundef %x, i32 noundef %n, ptr addrspace(1) nocapture noundef writeonly %ret) local_unnamed_addr {
entry:
  %call = call spir_func <3 x float> @_Z5rootnDv3_fi(<3 x float> noundef %x, i32 noundef %n)
  %extractVec2 = shufflevector <3 x float> %call, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %storetmp3 = bitcast ptr addrspace(1) %ret to ptr addrspace(1)
  store <4 x float> %extractVec2, ptr addrspace(1) %storetmp3, align 16
  ret void
}

declare spir_func <3 x float> @_Z5rootnDv3_fi(<3 x float> noundef, i32 noundef) local_unnamed_addr

;; __kernel void test_kernel_double(double3 x, int n, __global double3* ret) {
;;    *ret = rootn(x, n);
;; }

; CHECK-SPIRV: %{{.*}} rootn

define dso_local spir_kernel void @test_kernel_double(<3 x double> noundef %x, i32 noundef %n, ptr addrspace(1) nocapture noundef writeonly %ret) local_unnamed_addr {
entry:
  %call = call spir_func <3 x double> @_Z5rootnDv3_di(<3 x double> noundef %x, i32 noundef %n)
  %extractVec2 = shufflevector <3 x double> %call, <3 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
  %storetmp3 = bitcast ptr addrspace(1) %ret to ptr addrspace(1)
  store <4 x double> %extractVec2, ptr addrspace(1) %storetmp3, align 32
  ret void
}

declare spir_func <3 x double> @_Z5rootnDv3_di(<3 x double> noundef, i32 noundef) local_unnamed_addr
