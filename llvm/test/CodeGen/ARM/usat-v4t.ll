; RUN: not llc -O1 -mtriple=armv4t-none-none-eabi %s -o - 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot select: intrinsic %llvm.arm.usat
define i32 @usat1() nounwind {
  %tmp = call i32 @llvm.arm.usat(i32 128, i32 31)
  ret i32 %tmp
}

declare i32 @llvm.arm.usat(i32, i32) nounwind readnone
