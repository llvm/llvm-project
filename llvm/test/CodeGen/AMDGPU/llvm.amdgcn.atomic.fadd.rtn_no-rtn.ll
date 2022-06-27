; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs -stop-after=amdgpu-isel < %s | FileCheck -enable-var-scope -check-prefixes=GFX11 %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs -stop-after=amdgpu-isel < %s | FileCheck -enable-var-scope -check-prefixes=GFX11 %s

; no-rtn

; GFX11: BUFFER_ATOMIC_ADD_F32_OFFEN
define amdgpu_ps void @raw_buffer_atomic_add_f32_noret__vgpr_val__sgpr_rsrc__vgpr_voffset_plus4095__sgpr_soffset(float %val, <4 x i32> inreg %rsrc, i32 %voffset, i32 inreg %soffset) {
  %voffset.add = add i32 %voffset, 4095
  %ret = call float @llvm.amdgcn.raw.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %voffset.add, i32 %soffset, i32 0)
  ret void
}

; GFX11: BUFFER_ATOMIC_ADD_F32_OFFEN
define amdgpu_ps void @raw_buffer_atomic_add_f32_noret__vgpr_val__sgpr_rsrc__vgpr_voffset__sgpr_soffset_slc(float %val, <4 x i32> inreg %rsrc, i32 %voffset, i32 inreg %soffset) {
  %ret = call float @llvm.amdgcn.raw.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %voffset, i32 %soffset, i32 2)
  ret void
}

; GFX11: BUFFER_ATOMIC_ADD_F32_OFFSET
define amdgpu_ps void @raw_buffer_atomic_add_f32_noret__vgpr_val__sgpr_rsrc__vgpr_voffset_4095__sgpr_soffset(float %val, <4 x i32> inreg %rsrc, i32 inreg %soffset) {
  %ret = call float @llvm.amdgcn.raw.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 4095, i32 %soffset, i32 0)
  ret void
}

; GFX11: BUFFER_ATOMIC_ADD_F32_IDXEN
define amdgpu_ps void @struct_buffer_atomic_add_f32_noret__vgpr_val__sgpr_rsrc__4095_voffset__sgpr_soffset(float %val, <4 x i32> inreg %rsrc, i32 %vindex, i32 inreg %soffset) {
  %ret = call float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %vindex, i32 4095, i32 %soffset, i32 0)
  ret void
}

; GFX11: BUFFER_ATOMIC_ADD_F32_IDXEN
define amdgpu_ps void @struct_buffer_atomic_add_f32_noret__vgpr_val__sgpr_rsrc__0_voffset__sgpr_soffset_slc(float %val, <4 x i32> inreg %rsrc, i32 %vindex, i32 inreg %soffset) {
  %ret = call float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %vindex, i32 0, i32 %soffset, i32 2)
  ret void
}

; GFX11: BUFFER_ATOMIC_ADD_F32_BOTHEN
define amdgpu_ps void @struct_buffer_atomic_add_f32_noret__vgpr_val__sgpr_rsrc__vgpr_voffset_plus4095__sgpr_soffset(float %val, <4 x i32> inreg %rsrc, i32 %vindex, i32 %voffset, i32 inreg %soffset) {
  %voffset.add = add i32 %voffset, 4095
  %ret = call float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %vindex, i32 %voffset.add, i32 %soffset, i32 0)
  ret void
}

; GFX11: BUFFER_ATOMIC_ADD_F32_BOTHEN
define amdgpu_ps void @xstruct_buffer_atomic_add_f32_noret__vgpr_val__sgpr_rsrc__vgpr_voffset__sgpr_soffset_slc(float %val, <4 x i32> inreg %rsrc, i32 %vindex, i32 %voffset, i32 inreg %soffset) {
  %ret = call float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset, i32 2)
  ret void
}


; rtn

; GFX11: BUFFER_ATOMIC_ADD_F32_OFFEN
define amdgpu_ps float @raw_buffer_atomic_add_f32_ret__vgpr_val__sgpr_rsrc__vgpr_voffset_plus4095__sgpr_soffset(float %val, <4 x i32> inreg %rsrc, i32 %voffset, i32 inreg %soffset) {
  %voffset.add = add i32 %voffset, 4095
  %ret = call float @llvm.amdgcn.raw.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %voffset.add, i32 %soffset, i32 0)
  ret float %ret
}

; GFX11: BUFFER_ATOMIC_ADD_F32_OFFEN
define amdgpu_ps float @raw_buffer_atomic_add_f32_ret__vgpr_val__sgpr_rsrc__vgpr_voffset__sgpr_soffset_slc(float %val, <4 x i32> inreg %rsrc, i32 %voffset, i32 inreg %soffset) {
  %ret = call float @llvm.amdgcn.raw.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %voffset, i32 %soffset, i32 2)
  ret float %ret
}

; GFX11: BUFFER_ATOMIC_ADD_F32_OFFSET
define amdgpu_ps float @raw_buffer_atomic_add_f32_ret__vgpr_val__sgpr_rsrc__vgpr_voffset_4095__sgpr_soffset(float %val, <4 x i32> inreg %rsrc, i32 inreg %soffset) {
  %ret = call float @llvm.amdgcn.raw.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 4095, i32 %soffset, i32 0)
  ret float %ret
}

; GFX11: BUFFER_ATOMIC_ADD_F32_IDXEN
define amdgpu_ps float @struct_buffer_atomic_add_f32_ret__vgpr_val__sgpr_rsrc__4095_voffset__sgpr_soffset(float %val, <4 x i32> inreg %rsrc, i32 %vindex, i32 inreg %soffset) {
  %ret = call float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %vindex, i32 4095, i32 %soffset, i32 0)
  ret float %ret
}

; GFX11: BUFFER_ATOMIC_ADD_F32_IDXEN
define amdgpu_ps float @struct_buffer_atomic_add_f32_ret__vgpr_val__sgpr_rsrc__0_voffset__sgpr_soffset_slc(float %val, <4 x i32> inreg %rsrc, i32 %vindex, i32 inreg %soffset) {
  %ret = call float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %vindex, i32 0, i32 %soffset, i32 2)
  ret float %ret
}

; GFX11: BUFFER_ATOMIC_ADD_F32_BOTHEN
define amdgpu_ps float @struct_buffer_atomic_add_f32_ret__vgpr_val__sgpr_rsrc__vgpr_voffset_plus4095__sgpr_soffset(float %val, <4 x i32> inreg %rsrc, i32 %vindex, i32 %voffset, i32 inreg %soffset) {
  %voffset.add = add i32 %voffset, 4095
  %ret = call float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %vindex, i32 %voffset.add, i32 %soffset, i32 0)
  ret float %ret
}

; GFX11: BUFFER_ATOMIC_ADD_F32_BOTHEN
define amdgpu_ps float @xstruct_buffer_atomic_add_f32_ret__vgpr_val__sgpr_rsrc__vgpr_voffset__sgpr_soffset_slc(float %val, <4 x i32> inreg %rsrc, i32 %vindex, i32 %voffset, i32 inreg %soffset) {
  %ret = call float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset, i32 2)
  ret float %ret
}

declare float @llvm.amdgcn.raw.buffer.atomic.fadd.f32(float, <4 x i32>, i32, i32, i32 immarg) #0
declare float @llvm.amdgcn.struct.buffer.atomic.fadd.f32(float, <4 x i32>, i32, i32, i32, i32 immarg) #0
attributes #0 = { nounwind }
