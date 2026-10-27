; Each AIR compute system value becomes AGX's special registers (or values
; derived from them): grids are whole threadgroups, SIMD-groups are 32
; threads. A scalar parameter takes the first component; narrower integers
; are truncated.
; RUN: opt -mtriple=agx -passes=agx-lower-air -S %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"

; CHECK-LABEL: define agx_kernel void @k(ptr addrspace(1) %out)
; CHECK: call i32 @llvm.agx.thread.position.in.grid.x()
; CHECK: call i32 @llvm.agx.threadgroup.position.in.grid.x()
; CHECK: call i32 @llvm.agx.thread.position.in.threadgroup.x()
; CHECK: call i32 @llvm.agx.threads.per.threadgroup.x()
; CHECK: call i32 @llvm.agx.threadgroups.per.grid.x()
; threads_per_grid
; CHECK: %[[G:[0-9]+]] = call i32 @llvm.agx.threadgroups.per.grid.x()
; CHECK: %[[T:[0-9]+]] = call i32 @llvm.agx.threads.per.threadgroup.x()
; CHECK: mul i32 %[[G]], %[[T]]
; CHECK: call i32 @llvm.agx.thread.index.in.threadgroup()
; CHECK: call i32 @llvm.agx.thread.index.in.simdgroup()
; CHECK: call i32 @llvm.agx.simdgroup.index.in.threadgroup()
; simdgroups_per_threadgroup: ceil(threads / 32)
; CHECK: %[[A:[0-9]+]] = add i32 %{{[0-9]+}}, 31
; CHECK: lshr i32 %[[A]], 5
; CHECK: trunc i32 %{{[0-9]+}} to i16
; threads_per_simdgroup is 32.
; CHECK: store i32 32, ptr addrspace(1)

define void @k(ptr addrspace(1) %out, <3 x i32> %tid, <3 x i32> %pid, <3 x i32> %tidtg, <3 x i32> %tgsize,
               <3 x i32> %numprog, <3 x i32> %gridsize, i32 %tgindex, i32 %simdlane, i32 %simdid,
               i32 %simdsize, i32 %numsimd, i16 %tid16) {
  %a = extractelement <3 x i32> %tid, i32 0
  %b = extractelement <3 x i32> %pid, i32 0
  %c = extractelement <3 x i32> %tidtg, i32 0
  %d = extractelement <3 x i32> %tgsize, i32 0
  %e = extractelement <3 x i32> %numprog, i32 0
  %f = extractelement <3 x i32> %gridsize, i32 0
  %s1 = add i32 %a, %b
  %s2 = add i32 %s1, %c
  %s3 = add i32 %s2, %d
  %s4 = add i32 %s3, %e
  %s5 = add i32 %s4, %f
  %s6 = add i32 %s5, %tgindex
  %s7 = add i32 %s6, %simdlane
  %s8 = add i32 %s7, %simdid
  %s9 = add i32 %s8, %numsimd
  %w = zext i16 %tid16 to i32
  %s10 = add i32 %s9, %w
  store i32 %s10, ptr addrspace(1) %out
  %p = getelementptr i32, ptr addrspace(1) %out, i32 1
  store i32 %simdsize, ptr addrspace(1) %p
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !{}, !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14}
!2 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
!3 = !{i32 1, !"air.thread_position_in_grid"}
!4 = !{i32 2, !"air.threadgroup_position_in_grid"}
!5 = !{i32 3, !"air.thread_position_in_threadgroup"}
!6 = !{i32 4, !"air.threads_per_threadgroup"}
!7 = !{i32 5, !"air.threadgroups_per_grid"}
!8 = !{i32 6, !"air.threads_per_grid"}
!9 = !{i32 7, !"air.thread_index_in_threadgroup"}
!10 = !{i32 8, !"air.thread_index_in_simdgroup"}
!11 = !{i32 9, !"air.simdgroup_index_in_threadgroup"}
!12 = !{i32 10, !"air.threads_per_simdgroup"}
!13 = !{i32 11, !"air.simdgroups_per_threadgroup"}
!14 = !{i32 12, !"air.thread_position_in_grid"}
