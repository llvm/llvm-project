; RUN: not llc < %s -mtriple=armv8-eabi 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=thumbv8-eabi 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot select: intrinsic %llvm.arm.stc2
define void @stc2(i8* %i) nounwind {
entry:
  call void @llvm.arm.stc2(i32 1, i32 2, i8* %i) nounwind
  ret void
}

declare void @llvm.arm.stc2(i32, i32, i8*) nounwind
