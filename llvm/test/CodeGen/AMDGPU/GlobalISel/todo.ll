; RUN: llc -mtriple=amdgcn-mesa-mesa3d -global-isel -global-isel-abort=2 %s -o - 2>&1 | FileCheck %s

; This isn't implemented, but we need to make sure we fall back to SelectionDAG
; instead of generating wrong code.
; CHECK: warning: Instruction selection used fallback path for non_void_ret
; CHECK: non_void_ret:
; CHECK-NOT: s_endpgm
define amdgpu_vs i32 @non_void_ret() {
  ret i32 0
}
