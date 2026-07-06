; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes='early-cse<memssa>' -earlycse-debug-hash < %s | FileCheck %s

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
