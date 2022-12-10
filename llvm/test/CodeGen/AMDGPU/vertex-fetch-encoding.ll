; RUN: llc < %s -march=r600 -show-mc-encoding -mcpu=cypress | FileCheck --check-prefix=EG --check-prefix=FUNC %s
; RUN: llc < %s -march=r600 -show-mc-encoding -mcpu=barts | FileCheck --check-prefix=EG --check-prefix=FUNC %s
; RUN: llc < %s -march=r600 -show-mc-encoding -mcpu=cayman | FileCheck --check-prefix=CM --check-prefix=FUNC %s

; FUNC-LABEL: {{^}}vtx_fetch32:
; EG: VTX_READ_32 T[[GPR:[0-9]]].X, T[[GPR]].X, 0, #1 ; encoding: [0x40,0x01,0x0[[GPR]],0x10,0x0[[GPR]],0xf0,0x5f,0x13,0x00,0x00,0x08,0x00
; CM: VTX_READ_32 T[[GPR:[0-9]]].X, T[[GPR]].X, 0, #1 ; encoding: [0x40,0x01,0x0[[GPR]],0x00,0x0[[GPR]],0xf0,0x5f,0x13,0x00,0x00,0x00,0x00

define amdgpu_kernel void @vtx_fetch32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %v = load i32, ptr addrspace(1) %in
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}vtx_fetch128:
; EG: VTX_READ_128 T[[DST:[0-9]]].XYZW, T[[SRC:[0-9]]].X, 0, #1 ; encoding: [0x40,0x01,0x0[[SRC]],0x40,0x0[[DST]],0x10,0x8d,0x18,0x00,0x00,0x08,0x00
; CM: VTX_READ_128 T[[DST:[0-9]]].XYZW, T[[SRC:[0-9]]].X, 0, #1 ; encoding: [0x40,0x01,0x0[[SRC]],0x00,0x0[[DST]],0x10,0x8d,0x18,0x00,0x00,0x00,0x00

define amdgpu_kernel void @vtx_fetch128(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %v = load <4 x i32>, ptr addrspace(1) %in
  store <4 x i32> %v, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}vtx_fetch32_id3:
; EG: VTX_READ_32 T[[GPR:[0-9]]].X, T[[GPR]].X, 0, #3 ; encoding: [0x40,0x03,0x0[[GPR]],0x10,0x0[[GPR]],0xf0,0x5f,0x13,0x00,0x00,0x08,0x00
; CM: VTX_READ_32 T[[GPR:[0-9]]].X, T[[GPR]].X, 0, #3 ; encoding: [0x40,0x03,0x0[[GPR]],0x00,0x0[[GPR]],0xf0,0x5f,0x13,0x00,0x00,0x00,0x00

define amdgpu_kernel void @vtx_fetch32_id3(ptr addrspace(1) %out, ptr addrspace(7) %in) {
  %v = load i32, ptr addrspace(7) %in
  store i32 %v, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}vtx_fetch32_id2:
; EG: VTX_READ_32 T[[GPR:[0-9]]].X, T[[GPR]].X, 0, #2 ; encoding: [0x40,0x02,0x0[[GPR]],0x10,0x0[[GPR]],0xf0,0x5f,0x13,0x00,0x00,0x08,0x00
; CM: VTX_READ_32 T[[GPR:[0-9]]].X, T[[GPR]].X, 0, #2 ; encoding: [0x40,0x02,0x0[[GPR]],0x00,0x0[[GPR]],0xf0,0x5f,0x13,0x00,0x00,0x00,0x00

@t = internal addrspace(4) constant [4 x i32] [i32 0, i32 1, i32 2, i32 3]

define amdgpu_kernel void @vtx_fetch32_id2(ptr addrspace(1) %out, i32 %in) {
  %a = getelementptr inbounds [4 x i32], ptr addrspace(4) @t, i32 0, i32 %in
  %v = load i32, ptr addrspace(4) %a
  store i32 %v, ptr addrspace(1) %out
  ret void
}
