; AIR's SIMD-group matrices become AGX's fragments: each thread holds the
; elements (r, c) and (r, c + 1), r = 4 * (t / 16) + (t / 2) % 4 and
; c = 4 * ((t / 8) % 2) + 2 * (t % 2). Loads and stores move those two
; elements at (c + j + origin.x) * stride.x + (r + origin.y) * stride.y,
; multiply-accumulate is llvm.agx.simdgroup.matrix.fma, and element-wise
; operations and phis apply to the fragment.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14g %s -o /dev/null

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: %[[T:.*]] = call i32 @llvm.agx.thread.index.in.simdgroup()
; CHECK: %matrix.row = zext i32 %{{.*}} to i64
; CHECK: %matrix.col = zext i32 %{{.*}} to i64
; CHECK: loop:
; CHECK: %acc = phi <2 x float> [ zeroinitializer, %entry ], [ %next, %loop ]
; CHECK: load float, ptr addrspace(1)
; CHECK: %next = call <2 x float> @llvm.agx.simdgroup.matrix.fma.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}}, <2 x float> %acc)
; CHECK: done:
; CHECK: %scaled = fmul <2 x float> %next, splat (float 2.000000e+00)
; CHECK: store float %{{.*}}, ptr addrspace(1)
; CHECK-NOT: <64 x float>

declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)
declare void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float>, ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)
declare <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float)

define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) {
entry:
  %zero = call <64 x float> @air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float 0.0)
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi <64 x float> [ %zero, %entry ], [ %next, %loop ]
  %ma = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1) %a, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  %mb = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1) %b, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 8, i64 1>, <2 x i64> zeroinitializer)
  %next = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %ma, <64 x float> %mb, <64 x float> %acc)
  %i.next = add i32 %i, 1
  %last = icmp eq i32 %i.next, 4
  br i1 %last, label %done, label %loop
done:
  %scaled = fmul <64 x float> %next, splat (float 2.0)
  call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %scaled, ptr addrspace(1) %c, <2 x i64> <i64 8, i64 8>, <2 x i64> <i64 1, i64 8>, <2 x i64> zeroinitializer)
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2, !3}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1}
!2 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1}
