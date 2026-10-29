; RUN: not llc < %s -mtriple=nvptx64-nvidia-cuda 2>&1 | FileCheck %s

; CHECK: error: <unknown>:0:0: llvm.clear_cache is not supported on this target.
define void @f(ptr %a, ptr %b) {
  call void @llvm.clear_cache(ptr %a, ptr %b)
  ret void
}
