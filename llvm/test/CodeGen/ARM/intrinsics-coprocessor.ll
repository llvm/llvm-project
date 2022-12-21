; RUN: llc < %s -mtriple=armv7-eabi -mcpu=cortex-a8 | FileCheck %s

define void @coproc(ptr %i) nounwind {
entry:
  ; CHECK: mrc p7, #1, r{{[0-9]+}}, c1, c1, #4
  %0 = tail call i32 @llvm.arm.mrc(i32 7, i32 1, i32 1, i32 1, i32 4) nounwind
  ; CHECK: mcr p7, #1, r{{[0-9]+}}, c1, c1, #4
  tail call void @llvm.arm.mcr(i32 7, i32 1, i32 %0, i32 1, i32 1, i32 4) nounwind
  ; CHECK: mrc2 p7, #1, r{{[0-9]+}}, c1, c1, #4
  %1 = tail call i32 @llvm.arm.mrc2(i32 7, i32 1, i32 1, i32 1, i32 4) nounwind
  ; CHECK: mcr2 p7, #1, r{{[0-9]+}}, c1, c1, #4
  tail call void @llvm.arm.mcr2(i32 7, i32 1, i32 %1, i32 1, i32 1, i32 4) nounwind
  ; CHECK: mcrr p7, #1, r{{[0-9]+}}, r{{[0-9]+}}, c1
  tail call void @llvm.arm.mcrr(i32 7, i32 1, i32 %0, i32 %1, i32 1) nounwind
  ; CHECK: mcrr2 p7, #1, r{{[0-9]+}}, r{{[0-9]+}}, c1
  tail call void @llvm.arm.mcrr2(i32 7, i32 1, i32 %0, i32 %1, i32 1) nounwind
  ; CHECK: cdp p7, #3, c1, c1, c1, #5
  tail call void @llvm.arm.cdp(i32 7, i32 3, i32 1, i32 1, i32 1, i32 5) nounwind
  ; CHECK: cdp2 p7, #3, c1, c1, c1, #5
  tail call void @llvm.arm.cdp2(i32 7, i32 3, i32 1, i32 1, i32 1, i32 5) nounwind
  ; CHECK: ldc p7, c3, [r{{[0-9]+}}]
  tail call void @llvm.arm.ldc(i32 7, i32 3, ptr %i) nounwind
  ; CHECK: ldcl p7, c3, [r{{[0-9]+}}]
  tail call void @llvm.arm.ldcl(i32 7, i32 3, ptr %i) nounwind
  ; CHECK: ldc2 p7, c3, [r{{[0-9]+}}]
  tail call void @llvm.arm.ldc2(i32 7, i32 3, ptr %i) nounwind
  ; CHECK: ldc2l p7, c3, [r{{[0-9]+}}]
  tail call void @llvm.arm.ldc2l(i32 7, i32 3, ptr %i) nounwind
  ; CHECK: stc p7, c3, [r{{[0-9]+}}]
  tail call void @llvm.arm.stc(i32 7, i32 3, ptr %i) nounwind
  ; CHECK: stcl p7, c3, [r{{[0-9]+}}]
  tail call void @llvm.arm.stcl(i32 7, i32 3, ptr %i) nounwind
  ; CHECK: stc2 p7, c3, [r{{[0-9]+}}]
  tail call void @llvm.arm.stc2(i32 7, i32 3, ptr %i) nounwind
  ; CHECK: stc2l p7, c3, [r{{[0-9]+}}]
  tail call void @llvm.arm.stc2l(i32 7, i32 3, ptr %i) nounwind
  ; CHECK: mrrc p1, #2, r{{[0-9]+}}, r{{[0-9]+}}, c3
  %2 = tail call { i32, i32 } @llvm.arm.mrrc(i32 1, i32 2, i32 3) nounwind
  ; CHECK: mrrc2 p1, #2, r{{[0-9]+}}, r{{[0-9]+}}, c3
  %3 = tail call { i32, i32 } @llvm.arm.mrrc2(i32 1, i32 2, i32 3) nounwind
  ret void
}

declare void @llvm.arm.ldc(i32, i32, ptr) nounwind

declare void @llvm.arm.ldcl(i32, i32, ptr) nounwind

declare void @llvm.arm.ldc2(i32, i32, ptr) nounwind

declare void @llvm.arm.ldc2l(i32, i32, ptr) nounwind

declare void @llvm.arm.stc(i32, i32, ptr) nounwind

declare void @llvm.arm.stcl(i32, i32, ptr) nounwind

declare void @llvm.arm.stc2(i32, i32, ptr) nounwind

declare void @llvm.arm.stc2l(i32, i32, ptr) nounwind

declare void @llvm.arm.cdp2(i32, i32, i32, i32, i32, i32) nounwind

declare void @llvm.arm.cdp(i32, i32, i32, i32, i32, i32) nounwind

declare void @llvm.arm.mcrr2(i32, i32, i32, i32, i32) nounwind

declare void @llvm.arm.mcrr(i32, i32, i32, i32, i32) nounwind

declare void @llvm.arm.mcr2(i32, i32, i32, i32, i32, i32) nounwind

declare i32 @llvm.arm.mrc2(i32, i32, i32, i32, i32) nounwind

declare void @llvm.arm.mcr(i32, i32, i32, i32, i32, i32) nounwind

declare i32 @llvm.arm.mrc(i32, i32, i32, i32, i32) nounwind

declare { i32, i32 } @llvm.arm.mrrc(i32, i32, i32) nounwind

declare { i32, i32 } @llvm.arm.mrrc2(i32, i32, i32) nounwind
