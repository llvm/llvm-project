; RUN: llc -march=amdgcn -mattr=-fp64-fp16-denormals -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-fp64-fp16-denormals,-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}madak_f16
; GCN: buffer_load_ushort v[[A_F16:[0-9]+]]
; GCN: buffer_load_ushort v[[B_F16:[0-9]+]]
; VI:  v_madak_f16_e32 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_F16]], 0x4900{{$}}
; VI:  buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define void @madak_f16(
    half addrspace(1)* %r,
    half addrspace(1)* %a,
    half addrspace(1)* %b) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b

  %t.val = fmul half %a.val, %b.val
  %r.val = fadd half %t.val, 10.0

  store half %r.val, half addrspace(1)* %r
  ret void
}

; GCN-LABEL: {{^}}madak_f16_use_2
; SI:  v_mad_f32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; SI:  v_mac_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI:  v_mad_f16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI:  v_mac_f16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: s_endpgm
define void @madak_f16_use_2(
    half addrspace(1)* %r0,
    half addrspace(1)* %r1,
    half addrspace(1)* %a,
    half addrspace(1)* %b,
    half addrspace(1)* %c) {
entry:
  %a.val = load half, half addrspace(1)* %a
  %b.val = load half, half addrspace(1)* %b
  %c.val = load half, half addrspace(1)* %c

  %t0.val = fmul half %a.val, %b.val
  %t1.val = fmul half %a.val, %c.val
  %r0.val = fadd half %t0.val, 10.0
  %r1.val = fadd half %t1.val, 10.0

  store half %r0.val, half addrspace(1)* %r0
  store half %r1.val, half addrspace(1)* %r1
  ret void
}
