; RUN: not llc < %s -O1 -mtriple=armv6-none-none-eabi 2>&1 | FileCheck %s

; immediate argument > upper-bound
; CHECK: LLVM ERROR: Cannot select: intrinsic %llvm.arm.usat
define i32 @usat1() nounwind {
  %tmp = call i32 @llvm.arm.usat(i32 128, i32 32)
  ret i32 %tmp
}

declare i32 @llvm.arm.usat(i32, i32) nounwind readnone
