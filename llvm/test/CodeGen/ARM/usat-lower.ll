; RUN: not llc < %s -O1 -mtriple=armv6-none-none-eabi 2>&1 | FileCheck %s

; immediate argument < lower-bound
; CHECK: LLVM ERROR: Cannot select: intrinsic %llvm.arm.usat
define i32 @usat1() nounwind {
  %tmp = call i32 @llvm.arm.usat(i32 128, i32 -1)
  ret i32 %tmp
}

declare i32 @llvm.arm.usat(i32, i32) nounwind readnone
