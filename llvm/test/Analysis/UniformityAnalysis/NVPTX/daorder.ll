; RUN: opt %s -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel i32 @daorder(i32 %n) {
; CHECK-LABEL: for function 'daorder'
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %cond = icmp slt i32 %tid, 0
  br i1 %cond, label %A, label %B ; divergent
; CHECK:  DIVERGENT: %cond =
; CHECK: DIVERGENT: br i1 %cond,
A:
  %defAtA = add i32 %n, 1 ; uniform
; CHECK-NOT: DIVERGENT: %defAtA =
  br label %C
B:
  %defAtB = add i32 %n, 2 ; uniform
; CHECK-NOT: DIVERGENT: %defAtB =
  br label %C
C:
  %defAtC = phi i32 [ %defAtA, %A ], [ %defAtB, %B ] ; divergent
; CHECK: DIVERGENT: %defAtC =
  br label %D

D:
  %i = phi i32 [0, %C], [ %i.inc, %E ] ; uniform
; CHECK-NOT: DIVERGENT: %i = phi
  br label %E

E:
  %i.inc = add i32 %i, 1
  %loopCnt = icmp slt i32 %i.inc, %n
; CHECK-NOT: DIVERGENT: %loopCnt =
  br i1 %loopCnt, label %D, label %exit

exit:
  ret i32 %n
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.laneid()
