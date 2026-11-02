; RUN: opt -S -passes='loop-mssa(loop-rotate),verify' %s | FileCheck %s

; The old header disappears. Keep the original identity space and the counts
; for the blocks whose executions have not changed.

define amdgpu_kernel void @_Z14divergent_loopPVi(ptr addrspace(1) %out) !wave.profile !0 {
; CHECK-LABEL: define amdgpu_kernel void @_Z14divergent_loopPVi(
; CHECK-SAME: !wave.profile [[PROFILE:![0-9]+]] {
; CHECK: entry:
; CHECK: br label %body, !wave.profile.block [[ENTRY:![0-9]+]]
; CHECK: exit:
; CHECK: ret void, !wave.profile.block [[EXIT:![0-9]+]]
; CHECK: body:
; CHECK: br i1 %cond, label %body, label %exit, !wave.profile.block [[BODY:![0-9]+]]
;
; CHECK: [[PROFILE]] = distinct !{i64 2, i64 2480672276464841217, i64 6, i64 30, i64 6, i64 24}
; CHECK: [[ENTRY]] = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 3}
; CHECK: [[EXIT]] = !{i64 2, i64 2480672276464841217, i64 2, i64 1}
; CHECK: [[BODY]] = !{i64 2, i64 2480672276464841217, i64 3, i64 1, i64 3, i64 2}
entry:
  br label %header, !wave.profile.block !1

header:
  %i = phi i32 [ 0, %entry ], [ %inc, %body ]
  %cond = icmp ult i32 %i, 4
  br i1 %cond, label %body, label %exit, !wave.profile.block !2

exit:
  ret void, !wave.profile.block !3

body:
  store volatile i32 %i, ptr addrspace(1) %out
  %inc = add nuw nsw i32 %i, 1
  br label %header, !wave.profile.block !4
}

!0 = !{i64 2, i64 2480672276464841217, i64 6, i64 30, i64 6, i64 24}
!1 = !{i64 2, i64 2480672276464841217, i64 0, i64 1, i64 1}
!2 = !{i64 2, i64 2480672276464841217, i64 1, i64 1, i64 3, i64 2}
!3 = !{i64 2, i64 2480672276464841217, i64 2, i64 1}
!4 = !{i64 2, i64 2480672276464841217, i64 3, i64 1, i64 1}
