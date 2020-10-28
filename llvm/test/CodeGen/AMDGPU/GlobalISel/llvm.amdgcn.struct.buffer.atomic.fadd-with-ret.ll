; RUN: not --crash llc -global-isel < %s -march=amdgcn -mcpu=gfx908 -verify-machineinstrs 2>&1 | FileCheck %s -check-prefix=GFX908

declare float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float, <4 x i32>, i32, i32, i32, i32 immarg) #0

; GFX908: error: {{.*}} return versions of fp atomics not supported

define amdgpu_ps float @buffer_atomic_add_f32_rtn(float %val, <4 x i32> inreg %rsrc, i32 %vindex, i32 %voffset, i32 inreg %soffset) {
main_body:
  %ret = call float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset, i32 0)
  ret float %ret
}
