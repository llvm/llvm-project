; RUN: not --crash llc -mtriple=amdgpu9.50-amd-amdhsa < %s

define amdgpu_kernel void @reduced_fabs_vector_truncate_crash(<2 x float> %0) {
  %2 = fptrunc <2 x float> %0 to <2 x half>
  %3 = shufflevector <2 x half> %2, <2 x half> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %4 = tail call <4 x half> @llvm.fabs.v4f16(<4 x half> %3)
  store <4 x half> %4, ptr addrspace(1) null, align 8
  ret void
}
