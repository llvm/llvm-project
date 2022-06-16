; RUN: llc -march=amdgcn -stop-after=amdgpu-isel < %s | FileCheck -enable-var-scope -check-prefixes=GCN-ISEL                %s

; RUN: llc -march=amdgcn -mcpu=verde   -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CISI    %s
; RUN: llc -march=amdgcn -mcpu=fiji    -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI      %s
; RUN: llc -march=amdgcn -mcpu=gfx900  -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9    %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10,GFX1010,GFX10W32 %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -mattr=+wavefrontsize32 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10,GFX1030,GFX10W32 %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -mattr=+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX10,GFX1030,GFX10W64 %s

; GCN-ISEL-LABEL: name:   sadd64rr
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0.entry:
; GCN-ISEL: S_ADD_U64_PSEUDO

; GCN-LABEL: @sadd64rr
; GCN:       s_add_u32
; GCN:       s_addc_u32
define amdgpu_kernel void @sadd64rr(i64 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %add = add i64 %a, %b
  store i64 %add, i64 addrspace(1)* %out
  ret void
}

; GCN-ISEL-LABEL: name:   sadd64ri
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0.entry:
; GCN-ISEL: S_ADD_U64_PSEUDO

; GCN-LABEL: @sadd64ri
; GCN:       s_add_u32  s{{[0-9]+}}, s{{[0-9]+}}, 0x56789876
; GCN:       s_addc_u32 s{{[0-9]+}}, s{{[0-9]+}}, 0x1234
define amdgpu_kernel void @sadd64ri(i64 addrspace(1)* %out, i64 %a) {
entry:
  %add = add i64 20015998343286, %a
  store i64 %add, i64 addrspace(1)* %out
  ret void
}

; GCN-ISEL-LABEL: name:   vadd64rr
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0.entry:
; GCN-ISEL: V_ADD_U64_PSEUDO

; GCN-LABEL: @vadd64rr
;
; CISI:	v_add_i32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; CISI:	v_addc_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
;
; VI:	v_add_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; VI:	v_addc_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
;
; GFX9:	v_add_co_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_addc_co_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
;
; GFX10W32: v_add_co_u32 v{{[0-9]+}}, [[CARRY:s[0-9]+]], s{{[0-9]+}}, v{{[0-9]+}}
; GFX10W64: v_add_co_u32 v{{[0-9]+}}, [[CARRY:s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, v{{[0-9]+}}
; GFX1010: v_add_co_ci_u32_e64 v{{[0-9]+}}, [[CARRY]], s{{[0-9]+}}, 0, [[CARRY]]
; GFX1030: v_add_co_ci_u32_e64 v{{[0-9]+}}, null, s{{[0-9]+}}, 0, [[CARRY]]
define amdgpu_kernel void @vadd64rr(i64 addrspace(1)* %out, i64 %a) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %add = add i64 %a, %tid.ext
  store i64 %add, i64 addrspace(1)* %out
  ret void
}

; GCN-ISEL-LABEL: name:   vadd64ri
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0.entry:
; GCN-ISEL: V_ADD_U64_PSEUDO

; GCN-LABEL: @vadd64ri
;
; CISI:	v_add_i32_e32 v0, vcc, 0x56789876, v0
; CISI:	v_mov_b32_e32 v1, 0x1234
; CISI: v_addc_u32_e32 v1, vcc, 0, v1, vcc
;
; VI: v_add_u32_e32 v0, vcc, 0x56789876, v0
; VI: v_mov_b32_e32 v1, 0x1234
; VI: v_addc_u32_e32 v1, vcc, 0, v1, vcc
;
; GFX9:	v_add_co_u32_e32 v0, vcc, 0x56789876, v0
; GFX9: v_mov_b32_e32 v1, 0x1234
; GFX9: v_addc_co_u32_e32 v1, vcc, 0, v1, vcc
;
; GFX10W32: v_add_co_u32 v{{[0-9]+}}, [[CARRY:s[0-9]+]], 0x56789876, v{{[0-9]+}}
; GFX10W64: v_add_co_u32 v{{[0-9]+}}, [[CARRY:s\[[0-9]+:[0-9]+\]]], 0x56789876, v{{[0-9]+}}
; GFX1010: v_add_co_ci_u32_e64 v{{[0-9]+}}, [[CARRY]], 0, 0x1234, [[CARRY]]
; GFX1030: v_add_co_ci_u32_e64 v{{[0-9]+}}, null, 0, 0x1234, [[CARRY]]
define amdgpu_kernel void @vadd64ri(i64 addrspace(1)* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %add = add i64 20015998343286, %tid.ext
  store i64 %add, i64 addrspace(1)* %out
  ret void
}

; GCN-ISEL-LABEL: name:   suaddo32
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0
; GCN-ISEL: S_ADD_I32
define amdgpu_kernel void @suaddo32(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 %a, i32 %b) #0 {
  %uadd = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %uadd, 0
  %carry = extractvalue { i32, i1 } %uadd, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}


; GCN-ISEL-LABEL: name:   uaddo32_vcc_user
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0
; GCN-ISEL: V_ADD_CO_U32_e64

; below we check selection to v_add/addc
; because the only user of VCC produced by the UADDOis v_cndmask.
; We select to VALU form to avoid unnecessary s_cselect to copy SCC to VCC

; GCN-LABEL: @uaddo32_vcc_user
;
; CISI:	v_add_i32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; CISI:	v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
;
; VI:	v_add_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; VI:	v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
;
; GFX9:	v_add_co_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; GFX9:	v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
;
; GFX10W32: v_add_co_u32 v{{[0-9]+}}, [[CARRY:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX10W64: v_add_co_u32 v{{[0-9]+}}, [[CARRY:s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX10: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, [[CARRY]]
define amdgpu_kernel void @uaddo32_vcc_user(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 %a, i32 %b) #0 {
  %uadd = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %uadd, 0
  %carry = extractvalue { i32, i1 } %uadd, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; GCN-ISEL-LABEL: name:   suaddo64
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0
; GCN-ISEL: S_ADD_U64_PSEUDO

; GCN-LABEL: @suaddo64
;
; GCN: s_add_u32
; GCN: s_addc_u32
define amdgpu_kernel void @suaddo64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 %a, i64 %b) #0 {
  %uadd = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue { i64, i1 } %uadd, 0
  %carry = extractvalue { i64, i1 } %uadd, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; GCN-ISEL-LABEL: name:   vuaddo64
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0
; GCN-ISEL: V_ADD_U64_PSEUDO

; GCN-LABEL: @vuaddo64
;
; CISI:	v_add_i32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v0
; CISI:	v_addc_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
;
; VI:	v_add_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v0
; VI:	v_addc_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
;
; GFX9:	v_add_co_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v0
; GFX9:	v_addc_co_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
;
; GFX10W32: v_add_co_u32 v{{[0-9]+}}, [[CARRY:s[0-9]+]], s{{[0-9]+}}, v0
; GFX10W64: v_add_co_u32 v{{[0-9]+}}, [[CARRY:s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, v0
; GFX1010: v_add_co_ci_u32_e64 v{{[0-9]+}}, [[CARRY]], s{{[0-9]+}}, 0, [[CARRY]]
; GFX1030: v_add_co_ci_u32_e64 v{{[0-9]+}}, null, s{{[0-9]+}}, 0, [[CARRY]]
define amdgpu_kernel void @vuaddo64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 %a) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %uadd = call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %a, i64 %tid.ext)
  %val = extractvalue { i64, i1 } %uadd, 0
  %carry = extractvalue { i64, i1 } %uadd, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; GCN-ISEL-LABEL: name:   ssub64rr
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0.entry:
; GCN-ISEL: S_SUB_U64_PSEUDO

; GCN-LABEL: @ssub64rr
; GCN:       s_sub_u32
; GCN:       s_subb_u32
define amdgpu_kernel void @ssub64rr(i64 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %sub = sub i64 %a, %b
  store i64 %sub, i64 addrspace(1)* %out
  ret void
}

; GCN-ISEL-LABEL: name:   ssub64ri
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0.entry:
; GCN-ISEL: S_SUB_U64_PSEUDO

; GCN-LABEL: @ssub64ri
; GCN:       s_sub_u32  s{{[0-9]+}}, 0x56789876, s{{[0-9]+}}
; GCN:       s_subb_u32 s{{[0-9]+}}, 0x1234, s{{[0-9]+}}
define amdgpu_kernel void @ssub64ri(i64 addrspace(1)* %out, i64 %a) {
entry:
  %sub = sub i64 20015998343286, %a
  store i64 %sub, i64 addrspace(1)* %out
  ret void
}

; GCN-ISEL-LABEL: name:   vsub64rr
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0.entry:
; GCN-ISEL: V_SUB_U64_PSEUDO

; GCN-LABEL: @vsub64rr
;
; CISI:	v_sub_i32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; CISI:	v_subbrev_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
;
; VI:	v_sub_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; VI:	v_subbrev_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
;
; GFX9:	v_sub_co_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_subbrev_co_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
;
; GFX10W32: v_sub_co_u32 v{{[0-9]+}}, [[CARRY:s[0-9]+]], s{{[0-9]+}}, v{{[0-9]+}}
; GFX10W64: v_sub_co_u32 v{{[0-9]+}}, [[CARRY:s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, v{{[0-9]+}}
; GFX1010: v_sub_co_ci_u32_e64 v{{[0-9]+}}, [[CARRY]], s{{[0-9]+}}, 0, [[CARRY]]
; GFX1030: v_sub_co_ci_u32_e64 v{{[0-9]+}}, null, s{{[0-9]+}}, 0, [[CARRY]]
define amdgpu_kernel void @vsub64rr(i64 addrspace(1)* %out, i64 %a) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %sub = sub i64 %a, %tid.ext
  store i64 %sub, i64 addrspace(1)* %out
  ret void
}

; GCN-ISEL-LABEL: name:   vsub64ri
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0.entry:
; GCN-ISEL: V_SUB_U64_PSEUDO

; GCN-LABEL: @vsub64ri
;
; CISI:	v_sub_i32_e32 v0, vcc, 0x56789876, v0
; CISI:	v_mov_b32_e32 v1, 0x1234
; CISI: v_subbrev_u32_e32 v1, vcc, 0, v1, vcc
;
; VI: v_sub_u32_e32 v0, vcc, 0x56789876, v0
; VI: v_mov_b32_e32 v1, 0x1234
; VI: v_subbrev_u32_e32 v1, vcc, 0, v1, vcc
;
; GFX9:	v_sub_co_u32_e32 v0, vcc, 0x56789876, v0
; GFX9: v_mov_b32_e32 v1, 0x1234
; GFX9: v_subbrev_co_u32_e32 v1, vcc, 0, v1, vcc
;
; GFX10W32: v_sub_co_u32 v{{[0-9]+}}, [[CARRY:s[0-9]+]], 0x56789876, v{{[0-9]+}}
; GFX10W64: v_sub_co_u32 v{{[0-9]+}}, [[CARRY:s\[[0-9]+:[0-9]+\]]], 0x56789876, v{{[0-9]+}}
; GFX1010: v_sub_co_ci_u32_e64 v{{[0-9]+}}, [[CARRY]], 0x1234, 0, [[CARRY]]
; GFX1030: v_sub_co_ci_u32_e64 v{{[0-9]+}}, null, 0x1234, 0, [[CARRY]]
define amdgpu_kernel void @vsub64ri(i64 addrspace(1)* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %sub = sub i64 20015998343286, %tid.ext
  store i64 %sub, i64 addrspace(1)* %out
  ret void
}

; GCN-ISEL-LABEL: name:   susubo32
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0
; GCN-ISEL: S_SUB_I32
define amdgpu_kernel void @susubo32(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 %a, i32 %b) #0 {
  %usub = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %usub, 0
  %carry = extractvalue { i32, i1 } %usub, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  ret void
}


; GCN-ISEL-LABEL: name:   usubo32_vcc_user
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0
; GCN-ISEL: V_SUB_CO_U32_e64

; below we check selection to v_sub/subb
; because the only user of VCC produced by the USUBOis v_cndmask.
; We select to VALU form to avoid unnecessary s_cselect to copy SCC to VCC

; GCN-LABEL: @usubo32_vcc_user
;
; CISI:	v_sub_i32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; CISI:	v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
;
; VI:	v_sub_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; VI:	v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
;
; GFX9:	v_sub_co_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v{{[0-9]+}}
; GFX9:	v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, vcc
;
; GFX10W32: v_sub_co_u32 v{{[0-9]+}}, [[CARRY:s[0-9]+]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX10W64: v_sub_co_u32 v{{[0-9]+}}, [[CARRY:s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, s{{[0-9]+}}
; GFX10: v_cndmask_b32_e64 v{{[0-9]+}}, 0, 1, [[CARRY]]
define amdgpu_kernel void @usubo32_vcc_user(i32 addrspace(1)* %out, i1 addrspace(1)* %carryout, i32 %a, i32 %b) #0 {
  %usub = call { i32, i1 } @llvm.usub.with.overflow.i32(i32 %a, i32 %b)
  %val = extractvalue { i32, i1 } %usub, 0
  %carry = extractvalue { i32, i1 } %usub, 1
  store i32 %val, i32 addrspace(1)* %out, align 4
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; GCN-ISEL-LABEL: name:   susubo64
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0
; GCN-ISEL: S_SUB_U64_PSEUDO

; GCN-LABEL: @susubo64
;
; GCN: s_sub_u32
; GCN: s_subb_u32
define amdgpu_kernel void @susubo64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 %a, i64 %b) #0 {
  %usub = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %a, i64 %b)
  %val = extractvalue { i64, i1 } %usub, 0
  %carry = extractvalue { i64, i1 } %usub, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; GCN-ISEL-LABEL: name:   vusubo64
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.0
; GCN-ISEL: V_SUB_U64_PSEUDO

; GCN-LABEL: @vusubo64
;
; CISI:	v_sub_i32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v0
; CISI:	v_subbrev_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
;
; VI:	v_sub_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v0
; VI:	v_subbrev_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
;
; GFX9:	v_sub_co_u32_e32 v{{[0-9]+}}, vcc, s{{[0-9]+}}, v0
; GFX9:	v_subbrev_co_u32_e32 v{{[0-9]+}}, vcc, 0, v{{[0-9]+}}, vcc
;
; GFX10W32: v_sub_co_u32 v{{[0-9]+}}, [[CARRY:s[0-9]+]], s{{[0-9]+}}, v0
; GFX10W64: v_sub_co_u32 v{{[0-9]+}}, [[CARRY:s\[[0-9]+:[0-9]+\]]], s{{[0-9]+}}, v0
; GFX1010: v_sub_co_ci_u32_e64 v{{[0-9]+}}, [[CARRY]], s{{[0-9]+}}, 0, [[CARRY]]
; GFX1030: v_sub_co_ci_u32_e64 v{{[0-9]+}}, null, s{{[0-9]+}}, 0, [[CARRY]]
define amdgpu_kernel void @vusubo64(i64 addrspace(1)* %out, i1 addrspace(1)* %carryout, i64 %a) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %usub = call { i64, i1 } @llvm.usub.with.overflow.i64(i64 %a, i64 %tid.ext)
  %val = extractvalue { i64, i1 } %usub, 0
  %carry = extractvalue { i64, i1 } %usub, 1
  store i64 %val, i64 addrspace(1)* %out, align 8
  store i1 %carry, i1 addrspace(1)* %carryout
  ret void
}

; GCN-ISEL-LABEL: name:   sudiv64
; GCN-ISEL-LABEL: body:
; GCN-ISEL-LABEL: bb.3
; GCN-ISEL: %[[CARRY:[0-9]+]]:sreg_64_xexec = V_ADD_CO_U32_e64
; GCN-ISEL: S_ADD_CO_PSEUDO %{{[0-9]+}}, killed %{{[0-9]+}}, killed %[[CARRY]]
; GCN-ISEL: %[[CARRY:[0-9]+]]:sreg_64_xexec = V_SUB_CO_U32_e64
; GCN-ISEL: S_SUB_CO_PSEUDO killed %{{[0-9]+}}, %{{[0-9]+}}, %[[CARRY]]
define amdgpu_kernel void @sudiv64(i64 addrspace(1)* %out, i64 %x, i64 %y) {
  %result = udiv i64 %x, %y
  store i64 %result, i64 addrspace(1)* %out
  ret void
}



declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64) #1

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) #1

declare { i64, i1 } @llvm.usub.with.overflow.i64(i64, i64) #1

declare { i32, i1 } @llvm.usub.with.overflow.i32(i32, i32) #1

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

