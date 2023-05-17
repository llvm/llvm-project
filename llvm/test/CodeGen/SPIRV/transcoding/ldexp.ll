;; Check that backend converts scalar arg to vector for ldexp math instructions

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; #pragma OPENCL EXTENSION cl_khr_fp16 : enable
;; #pragma OPENCL EXTENSION cl_khr_fp64 : enable

;; __kernel void test_kernel_half(half3 x, int k, __global half3* ret) {
;;    *ret = ldexp(x, k);
;; }

; CHECK-SPIRV: %{{.*}} ldexp

define dso_local spir_kernel void @test_kernel_half(<3 x half> noundef %x, i32 noundef %k, <3 x half> addrspace(1)* nocapture noundef writeonly %ret) local_unnamed_addr {
entry:
  %call = call spir_func <3 x half> @_Z5ldexpDv3_Dhi(<3 x half> noundef %x, i32 noundef %k)
  %extractVec2 = shufflevector <3 x half> %call, <3 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  %storetmp3 = bitcast <3 x half> addrspace(1)* %ret to <4 x half> addrspace(1)*
  store <4 x half> %extractVec2, <4 x half> addrspace(1)* %storetmp3, align 8
  ret void
}

declare spir_func <3 x half> @_Z5ldexpDv3_Dhi(<3 x half> noundef, i32 noundef) local_unnamed_addr

;; __kernel void test_kernel_float(float3 x, int k, __global float3* ret) {
;;    *ret = ldexp(x, k);
;; }

; CHECK-SPIRV: %{{.*}} ldexp

define dso_local spir_kernel void @test_kernel_float(<3 x float> noundef %x, i32 noundef %k, <3 x float> addrspace(1)* nocapture noundef writeonly %ret) local_unnamed_addr {
entry:
  %call = call spir_func <3 x float> @_Z5ldexpDv3_fi(<3 x float> noundef %x, i32 noundef %k)
  %extractVec2 = shufflevector <3 x float> %call, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  %storetmp3 = bitcast <3 x float> addrspace(1)* %ret to <4 x float> addrspace(1)*
  store <4 x float> %extractVec2, <4 x float> addrspace(1)* %storetmp3, align 16
  ret void
}

declare spir_func <3 x float> @_Z5ldexpDv3_fi(<3 x float> noundef, i32 noundef) local_unnamed_addr

;; __kernel void test_kernel_double(double3 x, int k, __global double3* ret) {
;;    *ret = ldexp(x, k);
;; }

; CHECK-SPIRV: %{{.*}} ldexp

define dso_local spir_kernel void @test_kernel_double(<3 x double> noundef %x, i32 noundef %k, <3 x double> addrspace(1)* nocapture noundef writeonly %ret) local_unnamed_addr {
entry:
  %call = call spir_func <3 x double> @_Z5ldexpDv3_di(<3 x double> noundef %x, i32 noundef %k)
  %extractVec2 = shufflevector <3 x double> %call, <3 x double> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  %storetmp3 = bitcast <3 x double> addrspace(1)* %ret to <4 x double> addrspace(1)*
  store <4 x double> %extractVec2, <4 x double> addrspace(1)* %storetmp3, align 32
  ret void
}

declare spir_func <3 x double> @_Z5ldexpDv3_di(<3 x double> noundef, i32 noundef) local_unnamed_addr
