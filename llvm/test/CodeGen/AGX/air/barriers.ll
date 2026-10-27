; A threadgroup barrier that orders memory becomes AGX's threadgroup barrier
; between threadgroup-scope release and acquire fences; one that orders no
; memory is the barrier alone. A SIMD-group runs in lockstep: its barrier is
; a threadgroup-scope fence, or nothing without memory flags.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

; CHECK-LABEL: define agx_kernel void @k(
; CHECK-NEXT: entry:
; CHECK-NEXT: fence syncscope("workgroup") release
; CHECK-NEXT: call void @llvm.agx.threadgroup.barrier()
; CHECK-NEXT: fence syncscope("workgroup") acquire
; CHECK-NEXT: call void @llvm.agx.threadgroup.barrier()
; CHECK-NEXT: fence syncscope("workgroup") acq_rel
; CHECK-NEXT: ret void

declare void @air.wg.barrier(i32, i32)
declare void @air.simdgroup.barrier(i32, i32)

define void @k(ptr addrspace(1) %p) {
  call void @air.wg.barrier(i32 2, i32 1)
  call void @air.wg.barrier(i32 0, i32 1)
  call void @air.simdgroup.barrier(i32 2, i32 1)
  call void @air.simdgroup.barrier(i32 0, i32 1)
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1}
