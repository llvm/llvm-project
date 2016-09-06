; RUN: opt  -mtriple amdgcn--- -analyze -divergence %s | FileCheck %s

; CHECK-LABEL: 'fs_interp'
; CHECK: DIVERGENT: %v = call float @llvm.SI.fs.interp(
define amdgpu_ps void @fs_interp(i32 inreg %prim_mask, <2 x i32> %interp_param) #1 {
  %v = call float @llvm.SI.fs.interp(i32 0, i32 0, i32 %prim_mask, <2 x i32> %interp_param)
  store volatile float %v, float addrspace(1)* undef
  ret void
}

; CHECK-LABEL: 'fs_constant'
; CHECK: DIVERGENT: %v = call float @llvm.SI.fs.constant(
define amdgpu_ps void @fs_constant(i32 inreg %prim_mask, <2 x i32> %interp_param) #1 {
  %v = call float @llvm.SI.fs.constant(i32 0, i32 0, i32 %prim_mask)
  store volatile float %v, float addrspace(1)* undef
  ret void
}

declare float @llvm.SI.fs.interp(i32, i32, i32, <2 x i32>) #0
declare float @llvm.SI.fs.constant(i32, i32, i32) #0

attributes #0 = { nounwind readnone }
