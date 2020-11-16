; RUN: llc -march=amdgcn -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX8 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX10 %s

; FIXME: GFX9 should be producing v_mad_u16 instead of v_mad_legacy_u16.

; GCN-LABEL: {{^}}mad_u16
; GCN: {{flat|global}}_load_ushort v[[A:[0-9]+]]
; GCN: {{flat|global}}_load_ushort v[[B:[0-9]+]]
; GCN: {{flat|global}}_load_ushort v[[C:[0-9]+]]
; GFX8: v_mad_u16 v[[R:[0-9]+]], v[[A]], v[[B]], v[[C]]
; GFX9: v_mad_legacy_u16 v[[R:[0-9]+]], v[[A]], v[[B]], v[[C]]
; GFX10: v_mad_u16 v[[R:[0-9]+]], v[[A]], v[[B]], v[[C]]
; GCN: {{flat|global}}_store_short v{{.+}}, v[[R]]
; GCN: s_endpgm
define amdgpu_kernel void @mad_u16(
    i16 addrspace(1)* %r,
    i16 addrspace(1)* %a,
    i16 addrspace(1)* %b,
    i16 addrspace(1)* %c) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %a.gep = getelementptr inbounds i16, i16 addrspace(1)* %a, i32 %tid
  %b.gep = getelementptr inbounds i16, i16 addrspace(1)* %b, i32 %tid
  %c.gep = getelementptr inbounds i16, i16 addrspace(1)* %c, i32 %tid

  %a.val = load volatile i16, i16 addrspace(1)* %a.gep
  %b.val = load volatile i16, i16 addrspace(1)* %b.gep
  %c.val = load volatile i16, i16 addrspace(1)* %c.gep

  %m.val = mul i16 %a.val, %b.val
  %r.val = add i16 %m.val, %c.val

  store i16 %r.val, i16 addrspace(1)* %r
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
