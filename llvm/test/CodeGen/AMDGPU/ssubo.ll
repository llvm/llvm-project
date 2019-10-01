; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs< %s | FileCheck -check-prefixes=GCN,SI,FUNC %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga -verify-machineinstrs< %s | FileCheck -check-prefixes=GCN,VI,FUNC %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=gfx900 -verify-machineinstrs< %s | FileCheck -check-prefixes=GCN,GFX9,FUNC %s


declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32) nounwind readnone
declare { i64, i1 } @llvm.ssub.with.overflow.i64(i64, i64) nounwind readnone
declare { <2 x i32>, <2 x i1> } @llvm.ssub.with.overflow.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

; FUNC-LABEL: {{^}}ssubo_i64_zext:
define amdgpu_kernel void @ssubo_i64_zext(i64 addrspace(1)* %out, i64 %a, i64 %b) nounwind {
  %ssub = call { i64, i1 } @llvm.ssub.with.overflow.i64(i64 %a, i64 %b) nounwind
  %val = extractvalue { i64, i1 } %ssub, 0
  %carry = extractvalue { i64, i1 } %ssub, 1
  %ext = zext i1 %carry to i64
  %add2 = add i64 %val, %ext
  store i64 %add2, i64 addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}s_ssubo_i32:
define amdgpu_kernel void @s_ssubo_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 %a, i32 %b) nounwind {
  %ssub = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %a, i32 %b) nounwind
  %val = extractvalue { i32, i1 } %ssub, 0
  %carry = extractvalue { i32, i1 } %ssub, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}v_ssubo_i32:
define amdgpu_kernel void @v_ssubo_i32(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 addrspace(1)* %aptr, i32 addrspace(1)* %bptr) nounwind {
  %a = load i32, i32 addrspace(1)* %aptr, align 4
  %b = load i32, i32 addrspace(1)* %bptr, align 4
  %ssub = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 %a, i32 %b) nounwind
  %val = extractvalue { i32, i1 } %ssub, 0
  %carry = extractvalue { i32, i1 } %ssub, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}s_ssubo_i64:
; GCN: s_sub_u32
; GCN: s_subb_u32
define amdgpu_kernel void @s_ssubo_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 %a, i64 %b) nounwind {
  %ssub = call { i64, i1 } @llvm.ssub.with.overflow.i64(i64 %a, i64 %b) nounwind
  %val = extractvalue { i64, i1 } %ssub, 0
  %carry = extractvalue { i64, i1 } %ssub, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}v_ssubo_i64:
; SI: v_sub_i32_e32 v{{[0-9]+}}, vcc,
; SI: v_subb_u32_e32 v{{[0-9]+}}, vcc,

; VI: v_sub_u32_e32 v{{[0-9]+}}, vcc,
; VI: v_subb_u32_e32 v{{[0-9]+}}, vcc,

; GFX9: v_sub_co_u32_e32 v{{[0-9]+}}, vcc,
; GFX9: v_subb_co_u32_e32 v{{[0-9]+}}, vcc,
define amdgpu_kernel void @v_ssubo_i64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 addrspace(1)* %aptr, i64 addrspace(1)* %bptr) nounwind {
  %a = load i64, i64 addrspace(1)* %aptr, align 4
  %b = load i64, i64 addrspace(1)* %bptr, align 4
  %ssub = call { i64, i1 } @llvm.ssub.with.overflow.i64(i64 %a, i64 %b) nounwind
  %val = extractvalue { i64, i1 } %ssub, 0
  %carry = extractvalue { i64, i1 } %ssub, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; FUNC-LABEL: {{^}}v_ssubo_v2i32:
; SICIVI: v_cmp_lt_i32
; SICIVI: v_cmp_lt_i32
; SICIVI: v_sub_{{[iu]}}32
; SICIVI: v_cmp_lt_i32
; SICIVI: v_cmp_lt_i32
; SICIVI: v_sub_{{[iu]}}32
define amdgpu_kernel void @v_ssubo_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %carryout, <2 x i32> addrspace(1)* %aptr, <2 x i32> addrspace(1)* %bptr) nounwind {
  %a = load <2 x i32>, <2 x i32> addrspace(1)* %aptr, align 4
  %b = load <2 x i32>, <2 x i32> addrspace(1)* %bptr, align 4
  %sadd = call { <2 x i32>, <2 x i1> } @llvm.ssub.with.overflow.v2i32(<2 x i32> %a, <2 x i32> %b) nounwind
  %val = extractvalue { <2 x i32>, <2 x i1> } %sadd, 0
  %carry = extractvalue { <2 x i32>, <2 x i1> } %sadd, 1
  store <2 x i32> %val, <2 x i32> addrspace(1)* %out, align 4
  %carry.ext = zext <2 x i1> %carry to <2 x i32>
  store <2 x i32> %carry.ext, <2 x i32> addrspace(1)* %carryout
  ret void
}
