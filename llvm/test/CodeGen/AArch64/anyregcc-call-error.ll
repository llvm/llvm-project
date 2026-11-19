; RUN: not llc -mtriple=aarch64-- -global-isel=0 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=aarch64-- -global-isel=1 -filetype=null %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: the anyregcc calling convention is only supported by the stackmap and patchpoint intrinsics
declare anyregcc void @anyregcc_func(i64)

define void @call_anyregcc_func(i64 %x) {
  call anyregcc void @anyregcc_func(i64 %x)
  ret void
}
