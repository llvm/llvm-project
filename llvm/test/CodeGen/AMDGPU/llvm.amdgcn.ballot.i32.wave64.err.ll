; RUN: not llc -global-isel=0 -mtriple=amdgpu9.00 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu9.00 -filetype=null %s 2>&1 | FileCheck %s

; An i32 ballot on a wave64 target cannot represent one bit per lane, so
; both SelectionDAG and GlobalISel must refuse to lower it instead of
; dropping the mask bits of the high lanes.

declare i32 @llvm.amdgcn.ballot.i32(i1)

; CHECK: error: {{.*}}ballot return type is narrower than the wavefront size
define amdgpu_cs i32 @ballot_i32_wave64(i32 %x, i32 %y) {
  %cmp = icmp eq i32 %x, %y
  %ballot = call i32 @llvm.amdgcn.ballot.i32(i1 %cmp)
  ret i32 %ballot
}
