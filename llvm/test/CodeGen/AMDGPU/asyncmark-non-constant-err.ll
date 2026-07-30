; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null %s 2>&1 | FileCheck %s -check-prefix=CHECK-GISEL

; The wait_asyncmark operand indexes compiler-tracked async mark state, so it
; must fold to a constant by instruction selection.

; CHECK: LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.wait.asyncmark
; CHECK-GISEL: LLVM ERROR: cannot select: G_INTRINSIC_W_SIDE_EFFECTS intrinsic(@llvm.amdgcn.wait.asyncmark), %10:sgpr(i16) (in function: non_constant_wait)

define amdgpu_kernel void @non_constant_wait(i16 %n) {
entry:
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 %n)
  ret void
}

declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16)
