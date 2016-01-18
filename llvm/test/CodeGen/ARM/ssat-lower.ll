; RUN: not llc < %s -O1 -mtriple=armv6-none-none-eabi 2>&1 | FileCheck %s

; immediate argument < lower-bound
; CHECK: LLVM ERROR: Cannot select: intrinsic %llvm.arm.ssat
define i32 @ssat1() nounwind {
  %tmp = call i32 @llvm.arm.ssat(i32 128, i32 0)
  ret i32 %tmp
}

declare i32 @llvm.arm.ssat(i32, i32) nounwind readnone
