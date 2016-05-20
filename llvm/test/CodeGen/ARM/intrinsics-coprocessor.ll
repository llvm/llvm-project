; RUN: llc < %s -mtriple=armv7-eabi -mcpu=cortex-a8 | FileCheck %s
; RUN: llc < %s -march=thumb -mtriple=thumbv7-eabi -mcpu=cortex-a8 | FileCheck %s

define void @coproc() nounwind {
entry:
  ; CHECK: mrc p7, #1, r0, c1, c1, #4
  %0 = tail call i32 @llvm.arm.mrc(i32 7, i32 1, i32 1, i32 1, i32 4) nounwind
  ; CHECK: mcr p7, #1, r0, c1, c1, #4
  tail call void @llvm.arm.mcr(i32 7, i32 1, i32 %0, i32 1, i32 1, i32 4) nounwind
  ; CHECK: mrc2 p7, #1, r1, c1, c1, #4
  %1 = tail call i32 @llvm.arm.mrc2(i32 7, i32 1, i32 1, i32 1, i32 4) nounwind
  ; CHECK: mcr2 p7, #1, r1, c1, c1, #4
  tail call void @llvm.arm.mcr2(i32 7, i32 1, i32 %1, i32 1, i32 1, i32 4) nounwind
  ; CHECK: mcrr p7, #1, r0, r1, c1
  tail call void @llvm.arm.mcrr(i32 7, i32 1, i32 %0, i32 %1, i32 1) nounwind
  ; CHECK: mcrr2 p7, #1, r0, r1, c1
  tail call void @llvm.arm.mcrr2(i32 7, i32 1, i32 %0, i32 %1, i32 1) nounwind
  ; CHECK: cdp p7, #3, c1, c1, c1, #5
  tail call void @llvm.arm.cdp(i32 7, i32 3, i32 1, i32 1, i32 1, i32 5) nounwind
  ; CHECK: cdp2 p7, #3, c1, c1, c1, #5
  tail call void @llvm.arm.cdp2(i32 7, i32 3, i32 1, i32 1, i32 1, i32 5) nounwind
  ret void
}

declare void @llvm.arm.cdp2(i32, i32, i32, i32, i32, i32) nounwind

declare void @llvm.arm.cdp(i32, i32, i32, i32, i32, i32) nounwind

declare void @llvm.arm.mcrr2(i32, i32, i32, i32, i32) nounwind

declare void @llvm.arm.mcrr(i32, i32, i32, i32, i32) nounwind

declare void @llvm.arm.mcr2(i32, i32, i32, i32, i32, i32) nounwind

declare i32 @llvm.arm.mrc2(i32, i32, i32, i32, i32) nounwind

declare void @llvm.arm.mcr(i32, i32, i32, i32, i32, i32) nounwind

declare i32 @llvm.arm.mrc(i32, i32, i32, i32, i32) nounwind
