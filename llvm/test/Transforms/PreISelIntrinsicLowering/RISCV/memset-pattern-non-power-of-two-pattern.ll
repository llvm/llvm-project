; RUN: not opt -mtriple=riscv64 -passes=pre-isel-intrinsic-lowering -S -o - %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Pattern width for memset_pattern must be a power of 2

define void @memset_pattern_i127_x(ptr %a, i127 %value, i64 %x) nounwind {
  tail call void @llvm.memset.pattern.p0.i64.i127(ptr %a, i127 %value, i64 %x, i1 0)
  ret void
}
