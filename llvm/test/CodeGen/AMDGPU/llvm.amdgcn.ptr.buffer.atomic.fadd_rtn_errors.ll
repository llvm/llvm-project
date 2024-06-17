; RUN: split-file %s %t

; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=null %t/raw-ret-f32-error.ll 2>&1 | FileCheck -check-prefix=ERR-RAW-F32-SDAG %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=null %t/struct-ret-f32-error.ll 2>&1 | FileCheck -check-prefix=ERR-STRUCT-F32-SDAG %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=null %t/raw-ret-v2f16-error.ll 2>&1 | FileCheck -check-prefix=ERR-RAW-V2F16-SDAG %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=null %t/struct-ret-v2f16-error.ll 2>&1 | FileCheck -check-prefix=ERR-STRUCT-V2F16-SDAG %s

; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=null %t/raw-ret-f32-error.ll 2>&1 | FileCheck -check-prefix=ERR-RAW-F32-GISEL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=null %t/struct-ret-f32-error.ll 2>&1 | FileCheck -check-prefix=ERR-STRUCT-F32-GISEL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=null %t/raw-ret-v2f16-error.ll 2>&1 | FileCheck -check-prefix=ERR-RAW-V2F16-GISEL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx908 -filetype=null %t/struct-ret-v2f16-error.ll 2>&1 | FileCheck -check-prefix=ERR-STRUCT-V2F16-GISEL %s

; Make sure buffer fadd atomics with return values are not selected
; for gfx908 where they do not work.

;--- raw-ret-f32-error.ll
; ERR-RAW-F32-SDAG: LLVM ERROR: Cannot select: {{.+}}: f32,ch = BUFFER_ATOMIC_FADD
; ERR-RAW-F32-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_AMDGPU_BUFFER_ATOMIC_FADD

define float @raw_ptr_buffer_atomic_fadd_f32_rtn(float %val, <4 x i32> inreg %rsrc, i32 inreg %soffset) {
  %ret = call float @llvm.amdgcn.raw.buffer.atomic.fadd.f32(float %val, <4 x i32> %rsrc, i32 0, i32 %soffset, i32 0)
  ret float %ret
}

;--- struct-ret-f32-error.ll
; ERR-STRUCT-F32-SDAG: LLVM ERROR: Cannot select: {{.+}}: f32,ch = BUFFER_ATOMIC_FADD
; ERR-STRUCT-F32-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(s32) = G_AMDGPU_BUFFER_ATOMIC_FADD

define float @struct_ptr_buffer_atomic_fadd_f32_rtn(float %val, ptr addrspace(8) inreg %rsrc, i32 %vindex, i32 %voffset, i32 inreg %soffset) {
  %ret = call float @llvm.amdgcn.struct.ptr.buffer.atomic.fadd.f32(float %val, ptr addrspace(8) %rsrc, i32 %vindex, i32 %voffset, i32 %soffset, i32 0)
  ret float %ret
}

;--- raw-ret-v2f16-error.ll
; ERR-RAW-V2F16-SDAG: LLVM ERROR: Cannot select: {{.+}}: v2f16,ch = BUFFER_ATOMIC_FADD
; ERR-RAW-V2F16-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(<2 x s16>) = G_AMDGPU_BUFFER_ATOMIC_FADD

define <2 x half> @raw_ptr_buffer_atomic_fadd_v2f16_rtn(<2 x half> %val, <4 x i32> inreg %rsrc, i32 inreg %soffset) {
  %ret = call <2 x half> @llvm.amdgcn.raw.buffer.atomic.fadd.v2f16(<2 x half> %val, <4 x i32> %rsrc, i32 0, i32 %soffset, i32 0)
  ret <2 x half> %ret
}

;--- struct-ret-v2f16-error.ll
; ERR-STRUCT-V2F16-SDAG: LLVM ERROR: Cannot select: {{.+}}: v2f16,ch = BUFFER_ATOMIC_FADD
; ERR-STRUCT-V2F16-GISEL: LLVM ERROR: cannot select: %{{[0-9]+}}:vgpr_32(<2 x s16>) = G_AMDGPU_BUFFER_ATOMIC_FADD

define <2 x half> @struct_ptr_buffer_atomic_fadd_v2f16_rtn(<2 x half> %val, ptr addrspace(8) inreg %rsrc, i32 %vindex, i32 %voffset, i32 inreg %soffset) {
  %ret = call <2 x half> @llvm.amdgcn.struct.ptr.buffer.atomic.fadd.v2f16(<2 x half> %val, ptr addrspace(8) %rsrc, i32 %vindex, i32 %voffset, i32 %soffset, i32 0)
  ret <2 x half> %ret
}


