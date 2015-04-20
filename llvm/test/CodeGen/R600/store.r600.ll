; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck --check-prefix=EG %s

; XXX: Merge this test into store.ll once it is supported on SI

; v4i32 store
; EG: {{^}}store_v4i32:
; EG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+\.XYZW, T[0-9]+\.X}}, 1

define void @store_v4i32(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %1 = load <4 x i32> addrspace(1) * %in
  store <4 x i32> %1, <4 x i32> addrspace(1)* %out
  ret void
}

; v4f32 store
; EG: {{^}}store_v4f32:
; EG: MEM_RAT_CACHELESS STORE_RAW T{{[0-9]+\.XYZW, T[0-9]+\.X}}, 1
define void @store_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %1 = load <4 x float> addrspace(1) * %in
  store <4 x float> %1, <4 x float> addrspace(1)* %out
  ret void
}
