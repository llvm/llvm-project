; RUN: llc -mtriple aarch64-none-linux-gnu -verify-machineinstrs -stop-after=prologepilog < %s | FileCheck %s

; Check that STATEPOINT instruction prefer to use sp in presense of fp.
target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare void @consume(i32 addrspace(1)* %obj)

define i1 @test(i32 addrspace(1)* %a) "frame-pointer"="all" gc "statepoint-example" {
entry:
  %safepoint_token = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 0, i32 0, i32 0) ["gc-live" (i32 addrspace(1)* %a)]
; CHECK: STATEPOINT 0, 0, 0, @return_i1, 2, 0, 2, 0, 2, 0, 2, 1, 1, 8, $sp, 24, 2, 0, 2, 1, 0, 0
  %call1 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token,  i32 0, i32 0)
  %call2 = call zeroext i1 @llvm.experimental.gc.result.i1(token %safepoint_token)
  call void @consume(i32 addrspace(1)* %call1)
  ret i1 %call2
}


declare i1 @return_i1()
declare token @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32)
declare i1 @llvm.experimental.gc.result.i1(token)
