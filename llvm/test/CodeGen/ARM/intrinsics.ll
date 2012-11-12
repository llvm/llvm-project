; RUN: llc < %s -mtriple=armv7-eabi -mcpu=cortex-a8 | FileCheck %s
; RUN: llc < %s -march=thumb -mtriple=thumbv7-eabi -mcpu=cortex-a8 | FileCheck %s

define void @coproc() nounwind {
entry:
  ; CHECK: mrc
  %0 = tail call i32 @llvm.arm.mrc(i32 7, i32 1, i32 1, i32 1, i32 4) nounwind
  ; CHECK: mcr
  tail call void @llvm.arm.mcr(i32 7, i32 1, i32 %0, i32 1, i32 1, i32 4) nounwind
  ; CHECK: mrc2
  %1 = tail call i32 @llvm.arm.mrc2(i32 7, i32 1, i32 1, i32 1, i32 4) nounwind
  ; CHECK: mcr2
  tail call void @llvm.arm.mcr2(i32 7, i32 1, i32 %1, i32 1, i32 1, i32 4) nounwind
  ; CHECK: mcrr
  tail call void @llvm.arm.mcrr(i32 7, i32 1, i32 %0, i32 %1, i32 1) nounwind
  ; CHECK: mcrr2
  tail call void @llvm.arm.mcrr2(i32 7, i32 1, i32 %0, i32 %1, i32 1) nounwind
  ; CHECK: cdp
  tail call void @llvm.arm.cdp(i32 7, i32 3, i32 1, i32 1, i32 1, i32 5) nounwind
  ; CHECK: cdp2
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
