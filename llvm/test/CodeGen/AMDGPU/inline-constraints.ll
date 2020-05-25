; RUN: not llc < %s -march=amdgcn -mcpu=bonaire -verify-machineinstrs | FileCheck --check-prefix=GCN %s
; RUN: not llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=GCN --check-prefix=VI %s

; RUN: not llc < %s -march=amdgcn -mcpu=bonaire -verify-machineinstrs 2>&1 | FileCheck --check-prefix=NOGCN --check-prefix=NOSI %s
; RUN: not llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs 2>&1 | FileCheck --check-prefix=NOGCN %s

; GCN-LABEL: {{^}}inline_reg_constraints:
; GCN: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GCN: flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GCN: flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GCN: flat_load_dwordx4 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GCN: flat_load_dwordx4 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GCN: s_load_dword s{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}]
; GCN: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}]
; GCN: s_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}]
; GCN: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}]
; GCN: s_load_dwordx4 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}]
; GCN: s_load_dwordx8 s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}]

define amdgpu_kernel void @inline_reg_constraints(i32 addrspace(1)* %ptr) {
entry:
  %v32 = tail call i32 asm sideeffect "flat_load_dword   $0, $1", "=v,v"(i32 addrspace(1)* %ptr)
  %v2_32 = tail call <2 x i32> asm sideeffect "flat_load_dwordx2 $0, $1", "=v,v"(i32 addrspace(1)* %ptr)
  %v64 =   tail call i64 asm sideeffect "flat_load_dwordx2 $0, $1", "=v,v"(i32 addrspace(1)* %ptr)
  %v4_32 = tail call <4 x i32> asm sideeffect "flat_load_dwordx4 $0, $1", "=v,v"(i32 addrspace(1)* %ptr)
  %v128 =  tail call i128 asm sideeffect "flat_load_dwordx4 $0, $1", "=v,v"(i32 addrspace(1)* %ptr)
  %s32 =   tail call i32 asm sideeffect "s_load_dword $0, $1", "=s,s"(i32 addrspace(1)* %ptr)
  %s32_2 = tail call <2 x i32> asm sideeffect "s_load_dwordx2 $0, $1", "=s,s"(i32 addrspace(1)* %ptr)
  %s64 =   tail call i64 asm sideeffect "s_load_dwordx2 $0, $1", "=s,s"(i32 addrspace(1)* %ptr)
  %s4_32 =  tail call <4 x i32> asm sideeffect "s_load_dwordx4 $0, $1", "=s,s"(i32 addrspace(1)* %ptr)
  %s128 =  tail call i128 asm sideeffect "s_load_dwordx4 $0, $1", "=s,s"(i32 addrspace(1)* %ptr)
  %s256 =  tail call <8 x i32> asm sideeffect "s_load_dwordx8 $0, $1", "=s,s"(i32 addrspace(1)* %ptr)
  ret void
}

; GCN-LABEL: {{^}}inline_sreg_constraint_m0:
; GCN: s_mov_b32 m0, -1
; GCN-NOT: m0
; GCN: ; use m0
define amdgpu_kernel void @inline_sreg_constraint_m0() {
  %m0 = tail call i32 asm sideeffect "s_mov_b32 m0, -1", "={m0}"()
  tail call void asm sideeffect "; use $0", "s"(i32 %m0)
  ret void
}

; GCN-LABEL: {{^}}inline_sreg_constraint_imm_i32:
; GCN: s_mov_b32 [[REG:s[0-9]+]], 32
; GCN: ; use [[REG]]
define amdgpu_kernel void @inline_sreg_constraint_imm_i32() {
  tail call void asm sideeffect "; use $0", "s"(i32 32)
  ret void
}

; GCN-LABEL: {{^}}inline_sreg_constraint_imm_f32:
; GCN: s_mov_b32 [[REG:s[0-9]+]], 1.0
; GCN: ; use [[REG]]
define amdgpu_kernel void @inline_sreg_constraint_imm_f32() {
  tail call void asm sideeffect "; use $0", "s"(float 1.0)
  ret void
}

; FIXME: Should be able to use s_mov_b64
; GCN-LABEL: {{^}}inline_sreg_constraint_imm_i64:
; GCN-DAG: s_mov_b32 s[[REG_LO:[0-9]+]], -4{{$}}
; GCN-DAG: s_mov_b32 s[[REG_HI:[0-9]+]], -1{{$}}
; GCN: ; use s{{\[}}[[REG_LO]]:[[REG_HI]]{{\]}}
define amdgpu_kernel void @inline_sreg_constraint_imm_i64() {
  tail call void asm sideeffect "; use $0", "s"(i64 -4)
  ret void
}

; GCN-LABEL: {{^}}inline_sreg_constraint_imm_f64:
; GCN-DAG: s_mov_b32 s[[REG_LO:[0-9]+]], 0{{$}}
; GCN-DAG: s_mov_b32 s[[REG_HI:[0-9]+]], 0x3ff00000{{$}}
; GCN: ; use s{{\[}}[[REG_LO]]:[[REG_HI]]{{\]}}
define amdgpu_kernel void @inline_sreg_constraint_imm_f64() {
  tail call void asm sideeffect "; use $0", "s"(double 1.0)
  ret void
}

;==============================================================================
; 'A' constraint, 16-bit operand
;==============================================================================

; NOSI: error: invalid operand for inline asm constraint 'A'
; VI-LABEL: {{^}}inline_A_constraint_H0:
; VI: v_mov_b32 {{v[0-9]+}}, 64
define i32 @inline_A_constraint_H0() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i16 64)
  ret i32 %v0
}

; NOSI: error: invalid operand for inline asm constraint 'A'
; VI-LABEL: {{^}}inline_A_constraint_H1:
; VI: v_mov_b32 {{v[0-9]+}}, -16
define i32 @inline_A_constraint_H1() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i16 -16)
  ret i32 %v0
}

; NOSI: error: invalid operand for inline asm constraint 'A'
; VI-LABEL: {{^}}inline_A_constraint_H2:
; VI: v_mov_b32 {{v[0-9]+}}, 0x3c00
define i32 @inline_A_constraint_H2() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i16 bitcast (half 1.0 to i16))
  ret i32 %v0
}

; NOSI: error: invalid operand for inline asm constraint 'A'
; VI-LABEL: {{^}}inline_A_constraint_H3:
; VI: v_mov_b32 {{v[0-9]+}}, 0xbc00
define i32 @inline_A_constraint_H3() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i16 bitcast (half -1.0 to i16))
  ret i32 %v0
}

; NOSI: error: invalid operand for inline asm constraint 'A'
; VI-LABEL: {{^}}inline_A_constraint_H4:
; VI: v_mov_b32 {{v[0-9]+}}, 0x3118
define i32 @inline_A_constraint_H4() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(half 0xH3118)
  ret i32 %v0
}

; NOSI: error: invalid operand for inline asm constraint 'A'
; VI-LABEL: {{^}}inline_A_constraint_H5:
; VI: v_mov_b32 {{v[0-9]+}}, 0x3118
define i32 @inline_A_constraint_H5() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i16 bitcast (half 0xH3118 to i16))
  ret i32 %v0
}

; NOSI: error: invalid operand for inline asm constraint 'A'
; VI-LABEL: {{^}}inline_A_constraint_H6:
; VI: v_mov_b32 {{v[0-9]+}}, 0xb800
define i32 @inline_A_constraint_H6() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(half -0.5)
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_H7() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i16 bitcast (half 0xH3119 to i16))
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_H8() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i16 bitcast (half 0xH3117 to i16))
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_H9() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i16 65)
  ret i32 %v0
}

;==============================================================================
; 'A' constraint, 32-bit operand
;==============================================================================

; GCN-LABEL: {{^}}inline_A_constraint_F0:
; GCN: v_mov_b32 {{v[0-9]+}}, -16
define i32 @inline_A_constraint_F0() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i32 -16)
  ret i32 %v0
}

; GCN-LABEL: {{^}}inline_A_constraint_F1:
; GCN: v_mov_b32 {{v[0-9]+}}, 1
define i32 @inline_A_constraint_F1() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i32 1)
  ret i32 %v0
}

; GCN-LABEL: {{^}}inline_A_constraint_F2:
; GCN: v_mov_b32 {{v[0-9]+}}, 0xbf000000
define i32 @inline_A_constraint_F2() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i32 bitcast (float -0.5 to i32))
  ret i32 %v0
}

; GCN-LABEL: {{^}}inline_A_constraint_F3:
; GCN: v_mov_b32 {{v[0-9]+}}, 0x40000000
define i32 @inline_A_constraint_F3() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i32 bitcast (float 2.0 to i32))
  ret i32 %v0
}

; GCN-LABEL: {{^}}inline_A_constraint_F4:
; GCN: v_mov_b32 {{v[0-9]+}}, 0xc0800000
define i32 @inline_A_constraint_F4() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(float -4.0)
  ret i32 %v0
}

; NOSI: error: invalid operand for inline asm constraint 'A'
; VI-LABEL: {{^}}inline_A_constraint_F5:
; VI: v_mov_b32 {{v[0-9]+}}, 0x3e22f983
define i32 @inline_A_constraint_F5() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i32 1042479491)
  ret i32 %v0
}

; GCN-LABEL: {{^}}inline_A_constraint_F6:
; GCN: v_mov_b32 {{v[0-9]+}}, 0x3f000000
define i32 @inline_A_constraint_F6() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(float 0.5)
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_F7() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i32 1042479490)
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_F8() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i32 -17)
  ret i32 %v0
}

;==============================================================================
; 'A' constraint, 64-bit operand
;==============================================================================

; GCN-LABEL: {{^}}inline_A_constraint_D0:
; GCN: v_mov_b32 {{v[0-9]+}}, -16
define i32 @inline_A_constraint_D0() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i64 -16)
  ret i32 %v0
}

; GCN-LABEL: {{^}}inline_A_constraint_D1:
; GCN: v_cvt_f32_f64 {{v[0-9]+}}, 0xc000000000000000
define i32 @inline_A_constraint_D1() {
  %v0 = tail call i32 asm "v_cvt_f32_f64 $0, $1", "=v,A"(i64 bitcast (double -2.0 to i64))
  ret i32 %v0
}

; GCN-LABEL: {{^}}inline_A_constraint_D2:
; GCN: v_cvt_f32_f64 {{v[0-9]+}}, 0x3fe0000000000000
define i32 @inline_A_constraint_D2() {
  %v0 = tail call i32 asm "v_cvt_f32_f64 $0, $1", "=v,A"(double 0.5)
  ret i32 %v0
}

; NOSI: error: invalid operand for inline asm constraint 'A'
; VI-LABEL: {{^}}inline_A_constraint_D3:
; VI: v_cvt_f32_f64 {{v[0-9]+}}, 0x3fc45f306dc9c882
define i32 @inline_A_constraint_D3() {
  %v0 = tail call i32 asm "v_cvt_f32_f64 $0, $1", "=v,A"(double 0.15915494309189532)
  ret i32 %v0
}

; NOSI: error: invalid operand for inline asm constraint 'A'
; VI-LABEL: {{^}}inline_A_constraint_D4:
; VI: v_cvt_f32_f64 {{v[0-9]+}}, 0x3fc45f306dc9c882
define i32 @inline_A_constraint_D4() {
  %v0 = tail call i32 asm "v_cvt_f32_f64 $0, $1", "=v,A"(i64 bitcast (double 0.15915494309189532 to i64))
  ret i32 %v0
}

; GCN-LABEL: {{^}}inline_A_constraint_D5:
; GCN: v_cvt_f32_f64 {{v[0-9]+}}, 0xc000000000000000
define i32 @inline_A_constraint_D5() {
  %v0 = tail call i32 asm "v_cvt_f32_f64 $0, $1", "=v,A"(double -2.0)
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_D8() {
  %v0 = tail call i32 asm "v_cvt_f32_f64 $0, $1", "=v,A"(double 1.1)
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_D9() {
  %v0 = tail call i32 asm "v_cvt_f32_f64 $0, $1", "=v,A"(i64 bitcast (double 0.1 to i64))
  ret i32 %v0
}

;==============================================================================
; 'A' constraint, v2x16 operand
;==============================================================================

; NOSI: error: invalid operand for inline asm constraint 'A'
; VI-LABEL: {{^}}inline_A_constraint_V0:
; VI: v_mov_b32 {{v[0-9]+}}, -4
define i32 @inline_A_constraint_V0() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(<2 x i16> <i16 -4, i16 -4>)
  ret i32 %v0
}

; NOSI: error: invalid operand for inline asm constraint 'A'
; VI-LABEL: {{^}}inline_A_constraint_V1:
; VI: v_mov_b32 {{v[0-9]+}}, 0xb800
define i32 @inline_A_constraint_V1() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(<2 x half> <half -0.5, half -0.5>)
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_V2() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(<2 x i16> <i16 -4, i16 undef>)
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_V3() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(<2 x half> <half undef, half -0.5>)
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_V4() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(<2 x i16> <i16 1, i16 2>)
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_V5() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(<4 x i16> <i16 0, i16 0, i16 0, i16 0>)
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_V6() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(<2 x i32> <i32 0, i32 0>)
  ret i32 %v0
}

;==============================================================================
; 'A' constraint, type errors
;==============================================================================

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_E1(i32 %x) {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i32 %x)
  ret i32 %v0
}

; NOGCN: error: invalid operand for inline asm constraint 'A'
define i32 @inline_A_constraint_E2() {
  %v0 = tail call i32 asm "v_mov_b32 $0, $1", "=v,A"(i128 100000000000000000000)
  ret i32 %v0
}
