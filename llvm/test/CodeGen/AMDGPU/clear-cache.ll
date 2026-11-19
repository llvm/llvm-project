; RUN: not llc < %s -mtriple=amdgpu9.00-amd-amdhsa 2>&1 | FileCheck %s

; CHECK: error: <unknown>:0:0: llvm.clear_cache is not supported on this target.
define void @f(ptr %a, ptr %b) {
  call void @llvm.clear_cache(ptr %a, ptr %b)
  ret void
}
