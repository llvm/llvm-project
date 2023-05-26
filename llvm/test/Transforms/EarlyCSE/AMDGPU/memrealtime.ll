; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes='early-cse<memssa>' -earlycse-debug-hash < %s | FileCheck %s
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; CHECK-LABEL: @memrealtime(
; CHECK: call i64 @llvm.amdgcn.s.memrealtime()
; CHECK: call i64 @llvm.amdgcn.s.memrealtime()
define amdgpu_kernel void @memrealtime(i64 %cycles) #0 {
entry:
  %0 = tail call i64 @llvm.amdgcn.s.memrealtime()
  %cmp3 = icmp sgt i64 %cycles, 0
  br i1 %cmp3, label %while.body, label %while.end

while.body:
  %1 = tail call i64 @llvm.amdgcn.s.memrealtime()
  %sub = sub nsw i64 %1, %0
  %cmp = icmp slt i64 %sub, %cycles
  br i1 %cmp, label %while.body, label %while.end

while.end:
  ret void
}

; CHECK-LABEL: @memtime(
; CHECK: call i64 @llvm.amdgcn.s.memtime()
; CHECK: call i64 @llvm.amdgcn.s.memtime()
define amdgpu_kernel void @memtime(i64 %cycles) #0 {
entry:
  %0 = tail call i64 @llvm.amdgcn.s.memtime()
  %cmp3 = icmp sgt i64 %cycles, 0
  br i1 %cmp3, label %while.body, label %while.end

while.body:
  %1 = tail call i64 @llvm.amdgcn.s.memtime()
  %sub = sub nsw i64 %1, %0
  %cmp = icmp slt i64 %sub, %cycles
  br i1 %cmp, label %while.body, label %while.end

while.end:
  ret void
}

declare i64 @llvm.amdgcn.s.memrealtime()
declare i64 @llvm.amdgcn.s.memtime()
