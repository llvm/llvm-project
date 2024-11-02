; RUN: opt < %s -S -mtriple=amdgcn-- -passes=early-cse -earlycse-debug-hash | FileCheck %s

; CHECK-LABEL: @no_cse
; CHECK: call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 0, i32 0)
; CHECK: call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 4, i32 0)
define void @no_cse(ptr addrspace(1) %out, <4 x i32> %in) {
  %a = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 0, i32 0)
  %b = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 4, i32 0)
  %c = add i32 %a, %b
  store i32 %c, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @cse_zero_offset
; CHECK: [[CSE:%[a-z0-9A-Z]+]] = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 0, i32 0)
; CHECK: add i32 [[CSE]], [[CSE]]
define void @cse_zero_offset(ptr addrspace(1) %out, <4 x i32> %in) {
  %a = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 0, i32 0)
  %b = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 0, i32 0)
  %c = add i32 %a, %b
  store i32 %c, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @cse_nonzero_offset
; CHECK: [[CSE:%[a-z0-9A-Z]+]] = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 4, i32 0)
; CHECK: add i32 [[CSE]], [[CSE]]
define void @cse_nonzero_offset(ptr addrspace(1) %out, <4 x i32> %in) {
  %a = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 4, i32 0)
  %b = call i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> %in, i32 4, i32 0)
  %c = add i32 %a, %b
  store i32 %c, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.s.buffer.load.i32(<4 x i32> nocapture, i32, i32)
