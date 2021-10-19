; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}inline_asm_input_v2i16:
; GCN: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @inline_asm_input_v2i16(i32 addrspace(1)* %out, <2 x i16> %in) #0 {
entry:
  %val = call i32 asm "s_mov_b32 $0, $1", "=r,r"(<2 x i16> %in) #0
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}inline_asm_input_v2f16:
; GCN: s_mov_b32 s0, s{{[0-9]+}}
define amdgpu_kernel void @inline_asm_input_v2f16(i32 addrspace(1)* %out, <2 x half> %in) #0 {
entry:
  %val = call i32 asm "s_mov_b32 $0, $1", "=r,r"(<2 x half> %in) #0
  store i32 %val, i32 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}inline_asm_output_v2i16:
; GCN: s_mov_b32 s{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @inline_asm_output_v2i16(<2 x i16> addrspace(1)* %out, i32 %in) #0 {
entry:
  %val = call <2 x i16> asm "s_mov_b32 $0, $1", "=r,r"(i32 %in) #0
  store <2 x i16> %val, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}inline_asm_output_v2f16:
; GCN: v_mov_b32 v{{[0-9]+}}, s{{[0-9]+}}
define amdgpu_kernel void @inline_asm_output_v2f16(<2 x half> addrspace(1)* %out, i32 %in) #0 {
entry:
  %val = call <2 x half> asm "v_mov_b32 $0, $1", "=v,r"(i32 %in) #0
  store <2 x half> %val, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}inline_asm_packed_v2i16:
; GCN: v_pk_add_u16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @inline_asm_packed_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> %in0, <2 x i16> %in1) #0 {
entry:
  %val = call <2 x i16> asm "v_pk_add_u16 $0, $1, $2", "=v,r,v"(<2 x i16> %in0, <2 x i16> %in1) #0
  store <2 x i16> %val, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}inline_asm_packed_v2f16:
; GCN: v_pk_add_f16 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @inline_asm_packed_v2f16(<2 x half> addrspace(1)* %out, <2 x half> %in0, <2 x half> %in1) #0 {
entry:
  %val = call <2 x half> asm "v_pk_add_f16 $0, $1, $2", "=v,r,v"(<2 x half> %in0, <2 x half> %in1) #0
  store <2 x half> %val, <2 x half> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
