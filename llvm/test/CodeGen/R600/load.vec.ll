; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG  %s
; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck --check-prefix=SI  %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=SI  %s

; load a v2i32 value from the global address space.
; EG: {{^}}load_v2i32:
; EG: VTX_READ_64 T{{[0-9]+}}.XY, T{{[0-9]+}}.X, 0
; SI: {{^}}load_v2i32:
; SI: buffer_load_dwordx2 v[{{[0-9]+:[0-9]+}}]
define void @load_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %in) {
  %a = load <2 x i32> addrspace(1) * %in
  store <2 x i32> %a, <2 x i32> addrspace(1)* %out
  ret void
}

; load a v4i32 value from the global address space.
; EG: {{^}}load_v4i32:
; EG: VTX_READ_128 T{{[0-9]+}}.XYZW, T{{[0-9]+}}.X, 0
; SI: {{^}}load_v4i32:
; SI: buffer_load_dwordx4 v[{{[0-9]+:[0-9]+}}]
define void @load_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %a = load <4 x i32> addrspace(1) * %in
  store <4 x i32> %a, <4 x i32> addrspace(1)* %out
  ret void
}
