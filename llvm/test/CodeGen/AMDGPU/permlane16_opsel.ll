; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1010 --stop-after=amdgpu-isel -verify-machineinstrs < %s | FileCheck -check-prefixes=SDAG,SDAG-GFX10 %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1010 --stop-after=instruction-select -verify-machineinstrs < %s | FileCheck -check-prefixes=GISEL %s
; RUN: llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1100 --stop-after=amdgpu-isel -verify-machineinstrs < %s | FileCheck -check-prefixes=SDAG,SDAG-GFX11 %s
; RUN: llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1100 --stop-after=instruction-select -verify-machineinstrs < %s | FileCheck -check-prefixes=GISEL %s

declare i32 @llvm.amdgcn.permlane16(i32, i32, i32, i32, i1, i1)
declare i32 @llvm.amdgcn.permlanex16(i32, i32, i32, i32, i1, i1)
declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workitem.id.y()

define amdgpu_kernel void @v_permlane16_b32_vss(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_vii(ptr addrspace(1) %out, i32 %src0) {
  ; SDAG: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 1, i32 2, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; FIXME-GFX10PLUS: It is allowed to have both immediates as literals
define amdgpu_kernel void @v_permlane16_b32_vll(ptr addrspace(1) %out, i32 %src0) {
  ; SDAG: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 4660, i32 49617, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_vvv(ptr addrspace(1) %out, i32 %src0) {
  ; SDAG-GFX10: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; SDAG-GFX11: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %tidy = call i32 @llvm.amdgcn.workitem.id.y()
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %tidx, i32 %tidy, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_vvs(ptr addrspace(1) %out, i32 %src0, i32 %src2) {
  ; SDAG: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %tidx, i32 %src2, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_vsv(ptr addrspace(1) %out, i32 %src0, i32 %src1) {
  ; SDAG-GFX10: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; SDAG-GFX11: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidy = call i32 @llvm.amdgcn.workitem.id.y()
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %tidy, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_vss_fi(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANE16_B32_e64 4, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 4, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 true, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_vss_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 4, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 4, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 true)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_vss_fi_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANE16_B32_e64 4, {{%[0-9]+}}, 4, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 4, {{%[0-9]+}}, 4, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %v = call i32 @llvm.amdgcn.permlane16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 true, i1 true)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_vss(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_vii(ptr addrspace(1) %out, i32 %src0) {
  ; SDAG: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 1, i32 2, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; FIXME-GFX10PLUS: It is allowed to have both immediates as literals
define amdgpu_kernel void @v_permlanex16_b32_vll(ptr addrspace(1) %out, i32 %src0) {
  ; SDAG: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 4660, i32 49617, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_vvv(ptr addrspace(1) %out, i32 %src0) {
  ; SDAG-GFX10: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; SDAG-GFX11: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %tidy = call i32 @llvm.amdgcn.workitem.id.y()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %tidx, i32 %tidy, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_vvs(ptr addrspace(1) %out, i32 %src0, i32 %src2) {
  ; SDAG: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %tidx, i32 %src2, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_vsv(ptr addrspace(1) %out, i32 %src0, i32 %src1) {
  ; SDAG-GFX10: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; SDAG-GFX11: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidy = call i32 @llvm.amdgcn.workitem.id.y()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %tidy, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_vss_fi(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANEX16_B32_e64 4, {{%[0-9]+}}, 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 4, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 true, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_vss_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 4, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 4, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 false, i1 true)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_vss_fi_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANEX16_B32_e64 4, {{%[0-9]+}}, 4, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 4, {{%[0-9]+}}, 4, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %src0, i32 %src0, i32 %src1, i32 %src2, i1 true, i1 true)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_tid_tid(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}(s32), 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}(s32), 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 %tidx, i32 %tidx, i32 %src1, i32 %src2, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_undef_tid(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}(s32), 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %undef = freeze i32 poison
  %v = call i32 @llvm.amdgcn.permlane16(i32 %undef, i32 %tidx, i32 %src1, i32 %src2, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_i_tid(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}(s32), 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlane16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_i_tid_fi(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANE16_B32_e64 4, {{%[0-9]+}}(s32), 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 4, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %undef = freeze i32 poison
  %v = call i32 @llvm.amdgcn.permlane16(i32 %undef, i32 %tidx, i32 %src1, i32 %src2, i1 true, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_i_tid_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}(s32), 4, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 0, {{%[0-9]+}}, 4, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %undef = freeze i32 poison
  %v = call i32 @llvm.amdgcn.permlane16(i32 %undef, i32 %tidx, i32 %src1, i32 %src2, i1 false, i1 true)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlane16_b32_i_tid_fi_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANE16_B32_e64 4, {{%[0-9]+}}(s32), 4, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANE16_B32_e64 4, {{%[0-9]+}}, 4, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %undef = freeze i32 poison
  %v = call i32 @llvm.amdgcn.permlane16(i32 %undef, i32 %tidx, i32 %src1, i32 %src2, i1 true, i1 true)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_tid_tid(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}(s32), 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}(s32), 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %tidx, i32 %tidx, i32 %src1, i32 %src2, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_undef_tid(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}(s32), 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %undef = freeze i32 poison
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %undef, i32 %tidx, i32 %src1, i32 %src2, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_i_tid(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}(s32), 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %v = call i32 @llvm.amdgcn.permlanex16(i32 12345, i32 %tidx, i32 %src1, i32 %src2, i1 false, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_i_tid_fi(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANEX16_B32_e64 4, {{%[0-9]+}}(s32), 0, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 4, {{%[0-9]+}}, 0, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %undef = freeze i32 poison
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %undef, i32 %tidx, i32 %src1, i32 %src2, i1 true, i1 false)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_i_tid_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}(s32), 4, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 0, {{%[0-9]+}}, 4, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %undef = freeze i32 poison
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %undef, i32 %tidx, i32 %src1, i32 %src2, i1 false, i1 true)
  store i32 %v, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @v_permlanex16_b32_i_tid_fi_bc(ptr addrspace(1) %out, i32 %src0, i32 %src1, i32 %src2) {
  ; SDAG: V_PERMLANEX16_B32_e64 4, {{%[0-9]+}}(s32), 4, killed {{%[0-9]+}}, 0, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  ; GISEL: V_PERMLANEX16_B32_e64 4, {{%[0-9]+}}, 4, {{%[0-9]+}}, 0, {{%[0-9]+}}, {{%[0-9]+}}, 0, implicit $exec
  %tidx = call i32 @llvm.amdgcn.workitem.id.x()
  %undef = freeze i32 poison
  %v = call i32 @llvm.amdgcn.permlanex16(i32 %undef, i32 %tidx, i32 %src1, i32 %src2, i1 true, i1 true)
  store i32 %v, ptr addrspace(1) %out
  ret void
}
