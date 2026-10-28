; AIR's SIMD-group and quad-group functions become AGX's intrinsics: shuffles
; take the lane as i32; reductions and scans take the operation (AGX::SIMDOp)
; by the element type and the signedness in the name; inclusive scans add the
; element to the hardware's exclusive scan; a ballot is widened to AIR's
; 64-bit mask, and vote_all compares it with the ballot of the active threads.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

; CHECK-LABEL: define agx_kernel void @k(
; CHECK: %[[LANE:.*]] = trunc i32 %{{.*}} to i16
; CHECK: %[[L:[0-9]+]] = zext i16 %[[LANE]] to i32
; CHECK-NEXT: %s = call float @llvm.agx.simd.shuffle.f32(float %f, i32 %[[L]])
; CHECK: %x = call i32 @llvm.agx.simd.shuffle.xor.i32(i32 %i, i32 5)
; CHECK: %u = call i32 @llvm.agx.quad.shuffle.down.i32(i32 %i, i32 1)
; CHECK: %fsum = call float @llvm.agx.simd.reduce.f32(float %f, i32 1)
; CHECK: %isum = call i32 @llvm.agx.simd.reduce.i32(i32 %i, i32 16)
; CHECK: %smin = call i32 @llvm.agx.simd.reduce.i32(i32 %i, i32 20)
; CHECK: %umax = call i32 @llvm.agx.simd.reduce.i32(i32 %i, i32 30)
; CHECK: %[[EX:.*]] = call i32 @llvm.agx.simd.prefix.i32(i32 %i, i32 16)
; CHECK: %incl = add i32 %[[EX]], %i
; CHECK: %excl = call float @llvm.agx.simd.prefix.f32(float %f, i32 1)
; CHECK: %[[B:.*]] = call i32 @llvm.agx.simd.ballot(i1 %c)
; CHECK: %ballot = zext i32 %[[B]] to i64
; CHECK: %any = icmp ne i64 %ballot, 0
; CHECK: %[[ACT:.*]] = call i32 @llvm.agx.simd.ballot(i1 true)
; CHECK: %[[ACT64:.*]] = zext i32 %[[ACT]] to i64
; CHECK: %all = icmp eq i64 %ballot, %[[ACT64]]
; CHECK: %[[Q:.*]] = call i32 @llvm.agx.quad.ballot(i1 %c)
; CHECK: %qb = trunc i32 %[[Q]] to i16

declare float @air.simd_shuffle.f32(float, i16)
declare i32 @air.simd_shuffle_xor.s.i32(i32, i16)
declare i32 @air.quad_shuffle_down.u.i32(i32, i16)
declare float @air.simd_sum.f32(float)
declare i32 @air.simd_sum.s.i32(i32)
declare i32 @air.simd_min.s.i32(i32)
declare i32 @air.simd_max.u.i32(i32)
declare i32 @air.simd_prefix_inclusive_sum.s.i32(i32)
declare float @air.simd_prefix_exclusive_sum.f32(float)
declare i64 @air.simd_ballot.i64(i1)
declare i1 @air.simd_vote_any.i64(i64)
declare i1 @air.simd_vote_all.i64(i64)
declare i16 @air.quad_ballot(i1)

define void @k(ptr addrspace(1) %out, i16 %lane) {
  %f = uitofp i16 %lane to float
  %i = zext i16 %lane to i32
  %c = icmp ult i16 %lane, 7
  %s = call float @air.simd_shuffle.f32(float %f, i16 %lane)
  %x = call i32 @air.simd_shuffle_xor.s.i32(i32 %i, i16 5)
  %u = call i32 @air.quad_shuffle_down.u.i32(i32 %i, i16 1)
  %fsum = call float @air.simd_sum.f32(float %f)
  %isum = call i32 @air.simd_sum.s.i32(i32 %i)
  %smin = call i32 @air.simd_min.s.i32(i32 %i)
  %umax = call i32 @air.simd_max.u.i32(i32 %i)
  %incl = call i32 @air.simd_prefix_inclusive_sum.s.i32(i32 %i)
  %excl = call float @air.simd_prefix_exclusive_sum.f32(float %f)
  %ballot = call i64 @air.simd_ballot.i64(i1 %c)
  %any = call i1 @air.simd_vote_any.i64(i64 %ballot)
  %all = call i1 @air.simd_vote_all.i64(i64 %ballot)
  %qb = call i16 @air.quad_ballot(i1 %c)
  %r1 = fadd float %s, %fsum
  %r2 = fadd float %r1, %excl
  store float %r2, ptr addrspace(1) %out
  %r3 = add i32 %x, %u
  %r4 = add i32 %r3, %isum
  %r5 = add i32 %r4, %smin
  %r6 = add i32 %r5, %umax
  %r7 = add i32 %r6, %incl
  %anyi = zext i1 %any to i32
  %alli = zext i1 %all to i32
  %qbi = zext i16 %qb to i32
  %r8 = add i32 %r7, %anyi
  %r9 = add i32 %r8, %alli
  %r10 = add i32 %r9, %qbi
  %p = getelementptr i32, ptr addrspace(1) %out, i32 1
  store i32 %r10, ptr addrspace(1) %p
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !{!1, !2}}
!1 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1}
!2 = !{i32 1, !"air.thread_index_in_simdgroup"}
