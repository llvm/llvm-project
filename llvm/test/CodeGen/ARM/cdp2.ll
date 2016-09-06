; RUN: not llc < %s -mtriple=armv7-eabi -mcpu=cortex-a8 2>&1 | FileCheck %s
; RUN: not llc < %s -march=thumb -mtriple=thumbv7-eabi -mcpu=cortex-a8 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Cannot select: intrinsic %llvm.arm.cdp2
define void @cdp2(i32 %a) #0 {
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %1 = load i32, i32* %a.addr, align 4
  call void @llvm.arm.cdp2(i32 %1, i32 2, i32 3, i32 4, i32 5, i32 6)
  ret void
}

declare void @llvm.arm.cdp2(i32, i32, i32, i32, i32, i32) nounwind
