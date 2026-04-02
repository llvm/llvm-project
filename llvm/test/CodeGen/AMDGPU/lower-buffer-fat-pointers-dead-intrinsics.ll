; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-lower-buffer-fat-pointers < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes=amdgpu-lower-buffer-fat-pointers < %s | FileCheck %s

; CHECK: @arbitrary
declare amdgpu_kernel void @arbitrary(ptr addrspace(1))

; COM: This used to cause verifier errors when "lowered"
declare <4 x i8> @llvm.masked.load.v4i8.p7(ptr addrspace(7) captures(none), i32 immarg, <4 x i1>, <4 x i8>)
; CHECK-NOT: llvm.masked.load
