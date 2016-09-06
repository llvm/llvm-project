; RUN: not llc -O1 -mtriple=armv4t-none-none-eabi %s -o - 2>&1 | FileCheck %s

; CHECK: Cannot select: intrinsic %llvm.arm.ssat
define i32 @ssat() nounwind {
  %tmp = call i32 @llvm.arm.ssat(i32 128, i32 1)
  ret i32 %tmp
}

declare i32 @llvm.arm.ssat(i32, i32) nounwind readnone
