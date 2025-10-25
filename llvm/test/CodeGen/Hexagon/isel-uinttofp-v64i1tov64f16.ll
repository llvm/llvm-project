; Tests the conversion pattern for v64i1 to v64f16
; r0, r3 and r9 registers are i32 types converted from
; v32i1 via a bitcasting sequence.

; RUN: llc -march=hexagon -mattr=+hvxv79,+hvx-length128b \
; RUN: %s -verify-machineinstrs -o - | FileCheck %s

; CHECK: [[V3:v[0-9]+]] = vsplat([[R0:r[0-9]+]])
; CHECK: [[Q0:q[0-9]+]] = vand([[V3]],[[R0]])
; CHECK: [[V4:v[0-9]+]].w = prefixsum([[Q0]])
; CHECK: [[V5:v[0-9]+]].w = vsub([[V4]].w,[[V3]].w)
; CHECK: [[V1:v[0-9]+]] = vsplat(r
; CHECK: [[V2:v[0-9]+]] = vsplat(r
; CHECK: [[V6:v[0-9]+]].w = vlsr([[V1]].w,[[V5]].w)
; CHECK: [[V7:v[0-9]+]].w = vlsr([[V2]].w,[[V5]].w)
; CHECK: [[V8:v[0-9]+]] = vand([[V6]],[[V3]])
; CHECK: [[V9:v[0-9]+]] = vand([[V7]],[[V3]])
; CHECK: [[V10:v[0-9]+]].h = vpacke([[V9]].w,[[V8]].w)
; CHECK: .hf = [[V10]].h

define <64 x half> @uitofp_i1(<64 x i16> %in0, <64 x i16> %in1)
{
   %in = icmp eq <64 x i16> %in0, %in1
   %fp0 = uitofp <64 x i1> %in to <64 x half>
   %out = fadd <64 x half> %fp0, %fp0
   ret <64 x half> %out
}
