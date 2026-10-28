; AIR lays out a <3 x float> in 16 bytes (v96:128:128), AGX in 12: before
; the module takes AGX's data layout, address computations over such types
; become byte offsets with AIR's strides and offsets, and stack slots byte
; arrays of AIR's size.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s
; (llc has given the module AGX's data layout before the pass)
; RUN: llc -mtriple=agx -mcpu=agx-g14g -stop-after=agx-lower-air %s -o - | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air"

%S = type { <3 x float>, float }

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: %[[I:.*]] = sext i32 %i to i64
; CHECK: %[[OFF:.*]] = mul i64 %[[I]], 16
; CHECK: %pin = getelementptr i8, ptr addrspace(1) %in, i64 %[[OFF]]
; CHECK: load <3 x float>, ptr addrspace(1) %pin, align 16
; CHECK: %[[J:.*]] = sext i32 %i to i64
; CHECK: %[[OFF2:.*]] = mul i64 %[[J]], 16
; CHECK: %pout = getelementptr i8, ptr addrspace(1) %out, i64 %[[OFF2]]
; CHECK: %pf = getelementptr inbounds i8, ptr addrspace(1) %s, i64 16
; CHECK: %slot = alloca [32 x i8], align 16, addrspace(5)
; CHECK: %pslot = getelementptr i8, ptr addrspace(5) %slot, i64 16
; float arrays keep their GEPs
; CHECK: %pg = getelementptr float, ptr addrspace(1) %out, i32 %i
define void @k(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(1) %s, <3 x i32> %tid) {
  %i = extractelement <3 x i32> %tid, i32 0
  %pin = getelementptr <3 x float>, ptr addrspace(1) %in, i32 %i
  %x = load <3 x float>, ptr addrspace(1) %pin, align 16
  %y = fmul <3 x float> %x, <float 2.0, float 3.0, float 0.5>
  %pout = getelementptr <3 x float>, ptr addrspace(1) %out, i32 %i
  store <3 x float> %y, ptr addrspace(1) %pout, align 16
  %pf = getelementptr inbounds %S, ptr addrspace(1) %s, i32 0, i32 1
  %f = load float, ptr addrspace(1) %pf
  %slot = alloca %S
  %pslot = getelementptr %S, ptr %slot, i32 0, i32 1
  store volatile float %f, ptr %pslot
  %g = load volatile float, ptr %pslot
  %pg = getelementptr float, ptr addrspace(1) %out, i32 %i
  store float %g, ptr addrspace(1) %pg
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float3", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float3", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 32, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"S", !"air.arg_name", !"s"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tid"}
