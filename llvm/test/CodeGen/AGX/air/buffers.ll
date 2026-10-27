; Device buffers stay pointers. A constant buffer is passed by reference only
; when the kernel just loads from it within the size !air.kernel gives it; a
; constant table read at a variable index stays a pointer. A structure that
; AIR describes by its size is held in bytes. The threadgroup buffer is the
; launch's dynamic threadgroup memory: a zero-sized threadgroup variable.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

; CHECK: @k.scratch = external addrspace(3) global [0 x i8], align 16
; CHECK-LABEL: define agx_kernel void @k(
; CHECK-SAME: ptr addrspace(1) %out,
; CHECK-SAME: ptr addrspace(4) byref(<4 x float>) %v,
; CHECK-SAME: ptr addrspace(4) %table,
; CHECK-SAME: ptr addrspace(4) byref([12 x i8]) %params)
; CHECK: store float %{{.*}}, ptr addrspace(3) @k.scratch
; CHECK: getelementptr float, ptr addrspace(4) %table, i32

define void @k(ptr addrspace(1) %out, ptr addrspace(2) %v, ptr addrspace(2) %table,
               ptr addrspace(3) %scratch, ptr addrspace(2) %params, i32 %i) {
  %vv = load <4 x float>, ptr addrspace(2) %v
  %e = extractelement <4 x float> %vv, i32 2
  store float %e, ptr addrspace(3) %scratch
  %pt = getelementptr float, ptr addrspace(2) %table, i32 %i
  %t = load float, ptr addrspace(2) %pt
  %p8 = getelementptr i8, ptr addrspace(2) %params, i32 8
  %q = load float, ptr addrspace(2) %p8
  %s = fadd float %t, %q
  store float %s, ptr addrspace(1) %out
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2, !3, !4, !5, !6}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"v"}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"table"}
!4 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"scratch"}
!5 = !{i32 4, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!6 = !{i32 5, !"air.thread_index_in_threadgroup"}
