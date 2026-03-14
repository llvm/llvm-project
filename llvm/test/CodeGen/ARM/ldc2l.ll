; RUN: not --crash llc < %s -mtriple=armv8-eabi 2>&1 | FileCheck %s
; RUN: not --crash llc < %s -mtriple=thumbv8-eabi 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot select: intrinsic %llvm.arm.ldc2l
define void @ldc2l(ptr %i) nounwind {
entry:
  call void @llvm.arm.ldc2l(i32 1, i32 2, ptr %i) nounwind
  ret void
}

declare void @llvm.arm.ldc2l(i32, i32, ptr) nounwind
