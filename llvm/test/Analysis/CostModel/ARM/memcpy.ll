; RUN: opt < %s  -cost-model -analyze -cost-kind=code-size | FileCheck %s

target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv7m-arm-unknown-eabi"

define void @memcpy(i8* %d, i8* %s, i32 %N) {
entry:
; CHECK: cost of 4 for instruction: call void @llvm.memcpy.p0i8.p0i8.i32
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %d, i8* align 1 %s, i32 36, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #1
