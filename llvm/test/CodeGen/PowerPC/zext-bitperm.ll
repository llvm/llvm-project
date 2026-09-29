; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-linux-gnu | FileCheck %s

; Test case for PPCTargetLowering::extendSubTreeForBitPermutation.
; We expect mask and rotate are folded into a rlwinm instruction.

define zeroext i32 @func(ptr %p, i32 zeroext %i) {
; CHECK-LABEL: @func
; CHECK: addi [[REG1:[0-9]+]], 4, 1
; CHECK: rlwinm [[REG2:[0-9]+]], [[REG1]], 2, 22, 29
; CHECK-NOT: sldi
; CHECK: lwzx 3, 3, [[REG2]]
; CHECK: blr
entry:
  %add = add i32 %i, 1
  %and = and i32 %add, 255
  %idxprom = zext i32 %and to i64
  %arrayidx = getelementptr inbounds i32, ptr %p, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

