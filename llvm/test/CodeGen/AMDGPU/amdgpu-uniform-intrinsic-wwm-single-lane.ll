; RUN: opt -mtriple=amdgcn--amdpal -amdgpu-uniform-intrinsic-combine -mattr=+wavefrontsize64 -S < %s | FileCheck %s -check-prefix=CHECK

; If uniformity analysis sees that the workgroup size is 1, it would say this function is trivially uniform.
; However, the function's use of wwm means that all lanes will be active even if the workgroup size is 1, so
; it should not be considered uniform, and we should not see the permlane64 optimized out.
define amdgpu_kernel void @kernel(i32 inreg noundef %input) #0 {
; CHECK-LABEL: define amdgpu_kernel void @kernel(
; CHECK:    call float @llvm.amdgcn.permlane64.f32
.entry:
  %0 = call i64 @llvm.amdgcn.s.getpc()
  %1 = and i64 %0, -4294967296
  %2 = zext i32 %input to i64
  %3 = or disjoint i64 %1, %2
  %4 = inttoptr i64 %3 to ptr addrspace(4)
  %5 = getelementptr i8, ptr addrspace(4) %4, i64 16
  %6 = load <4 x i32>, ptr addrspace(4) %5, align 16
  %7 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %8 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %7)
  %9 = shl i32 %8, 2
  %10 = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %6, i32 %9, i32 0)
  %11 = call i32 @llvm.amdgcn.set.inactive.i32(i32 %10, i32 0)
  %12 = bitcast i32 %11 to float
  %13 = call float @llvm.amdgcn.permlane64.f32(float %12)
  %14 = fadd float %13, %12
  %15 = call float @llvm.amdgcn.readfirstlane.f32(float %14)
  %16 = bitcast float %15 to i32
  %17 = call i32 @llvm.amdgcn.strict.wwm.i32(i32 %16)
  ret void
}

attributes #0 = { alwaysinline nounwind memory(readwrite) "amdgpu-flat-work-group-size"="1,1" }
