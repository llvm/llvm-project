; RUN: llc -mtriple=amdgcn-amd-amdhsa -global-isel < %s
; LLVM ERROR: cannot select: G_UBSANTRAP 0 (in function: ubsan_trap)

define void @ubsan_trap() {
  call void @llvm.ubsantrap(i8 0)
  ret void
}