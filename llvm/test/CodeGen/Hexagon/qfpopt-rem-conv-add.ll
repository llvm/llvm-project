; Tests if generated vadd instruction takes in qf32
; type as first parameter instead of a sf type without
; any conversion instruction of type sf = qf32

; RUN: llc -mtriple=hexagon -mattr=+hvx-length128b,+hvxv75,+v75 < %s -o - | FileCheck %s

; CHECK: [[V2:v[0-9]+]] = vxor([[V2]],[[V2]])
; CHECK: [[V0:v[0-9]+]].qf32 = vmpy([[V0]].sf,[[V2]].sf)
; CHECK: [[V1:v[0-9]+]].qf32 = vmpy([[V1]].sf,[[V2]].sf)
; CHECK: [[V4:v[0-9]+]].qf32 = vadd([[V0]].qf32,[[V2]].sf)
; CHECK: [[V5:v[0-9]+]].qf32 = vadd([[V1]].qf32,[[V2]].sf)

define void @_Z19compute_ripple_geluIDF16_EviPT_PKS0_(ptr %out_ptr, <64 x float> %conv14.ripple.vectorized) #0 {
entry:
  %mul16.ripple.vectorized = fmul <64 x float> %conv14.ripple.vectorized, zeroinitializer
  %conv17.ripple.vectorized = fptrunc <64 x float> %mul16.ripple.vectorized to <64 x half>
  store <64 x half> %conv17.ripple.vectorized, ptr %out_ptr, align 2
  ret void
}
