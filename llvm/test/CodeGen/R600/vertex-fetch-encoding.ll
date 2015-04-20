; RUN: llc < %s -march=r600 -show-mc-encoding -mcpu=barts | FileCheck --check-prefix=NI %s
; RUN: llc < %s -march=r600 -show-mc-encoding -mcpu=cayman | FileCheck --check-prefix=CM %s

; NI: {{^}}vtx_fetch32:
; NI: VTX_READ_32 T[[GPR:[0-9]]].X, T[[GPR]].X, 0 ; encoding: [0x40,0x01,0x0[[GPR]],0x10,0x0[[GPR]],0xf0,0x5f,0x13,0x00,0x00,0x08,0x00
; CM: {{^}}vtx_fetch32:
; CM: VTX_READ_32 T[[GPR:[0-9]]].X, T[[GPR]].X, 0 ; encoding: [0x40,0x01,0x0[[GPR]],0x00,0x0[[GPR]],0xf0,0x5f,0x13,0x00,0x00,0x00,0x00

define void @vtx_fetch32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
entry:
  %0 = load i32 addrspace(1)* %in
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; NI: {{^}}vtx_fetch128:
; NI: VTX_READ_128 T[[DST:[0-9]]].XYZW, T[[SRC:[0-9]]].X, 0 ; encoding: [0x40,0x01,0x0[[SRC]],0x40,0x0[[DST]],0x10,0x8d,0x18,0x00,0x00,0x08,0x00
; XXX: Add a case for Cayman when v4i32 stores are supported.

define void @vtx_fetch128(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
entry:
  %0 = load <4 x i32> addrspace(1)* %in
  store <4 x i32> %0, <4 x i32> addrspace(1)* %out
  ret void
}
