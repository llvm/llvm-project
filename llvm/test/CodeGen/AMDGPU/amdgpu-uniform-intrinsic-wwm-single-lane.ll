; RUN: opt -mtriple=amdgcn--amdpal -amdgpu-uniform-intrinsic-combine -mattr=+wavefrontsize64 -S < %s | FileCheck %s -check-prefix=CHECK

; If uniformity analysis sees that the workgroup size is 1, it would say this function is trivially uniform.
; However, the function's use of wwm means that all lanes will be active even if the workgroup size is 1, so
; it should not be considered uniform, and we should not see the permlane64 optimized out.
define amdgpu_kernel void @kernel(ptr addrspace(4) %input, ptr addrspace(1) %out) #0 {
; CHECK-LABEL: define amdgpu_kernel void @kernel(
; CHECK:    call float @llvm.amdgcn.permlane64.f32
.entry:
  %element_ptr = getelementptr i8, ptr addrspace(4) %input, i64 16
  %buffer = load <4 x i32>, ptr addrspace(4) %element_ptr, align 16
  %lane_id_lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %lane_id = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane_id_lo)
  %lane_id_idx = shl i32 %lane_id, 2
  %vals = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %buffer, i32 %lane_id_idx, i32 0)
  %vals_inactive_zeroed = call i32 @llvm.amdgcn.set.inactive.i32(i32 %vals, i32 0)
  %float_vals = bitcast i32 %vals_inactive_zeroed to float
  %swapped_vals = call float @llvm.amdgcn.permlane64.f32(float %float_vals)
  %sum = fadd float %swapped_vals, %float_vals
  %res = call float @llvm.amdgcn.readfirstlane.f32(float %sum)
  %res_i32 = bitcast float %res to i32
  %res_i32_wwm = call i32 @llvm.amdgcn.strict.wwm.i32(i32 %res_i32)
  store i32 %res_i32_wwm, ptr addrspace(1) %out
  ret void
}

attributes #0 = { alwaysinline nounwind memory(readwrite) "amdgpu-flat-work-group-size"="1,1" }
