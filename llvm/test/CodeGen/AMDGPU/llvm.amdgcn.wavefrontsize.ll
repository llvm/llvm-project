; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,W64 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,W32 %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,W64 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -mattr=+wavefrontsize32,-wavefrontsize64 -verify-machineinstrs -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,W32 %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,W64 %s

; RUN: opt -O3 -S < %s | FileCheck -check-prefixes=OPT,OPT-WXX %s
; RUN: opt -mtriple=amdgcn-- -O3 -S < %s | FileCheck -check-prefixes=OPT,OPT-WXX %s
; RUN: opt -mtriple=amdgcn-- -O3 -mattr=+wavefrontsize32 -S < %s | FileCheck -check-prefixes=OPT,OPT-W32 %s
; RUN: opt -mtriple=amdgcn-- -passes='default<O3>' -mattr=+wavefrontsize32 -S < %s | FileCheck -check-prefixes=OPT,OPT-W32 %s
; RUN: opt -mtriple=amdgcn-- -O3 -mattr=+wavefrontsize64 -S < %s | FileCheck -check-prefixes=OPT,OPT-W64 %s
; RUN: opt -mtriple=amdgcn-- -mcpu=tonga -O3 -S < %s | FileCheck -check-prefixes=OPT,OPT-W64 %s
; RUN: opt -mtriple=amdgcn-- -mcpu=gfx1010 -O3 -mattr=+wavefrontsize32,-wavefrontsize64 -S < %s | FileCheck -check-prefixes=OPT,OPT-W32 %s
; RUN: opt -mtriple=amdgcn-- -mcpu=gfx1010 -O3 -mattr=-wavefrontsize32,+wavefrontsize64 -S < %s | FileCheck -check-prefixes=OPT,OPT-W64 %s
; RUN: opt -mtriple=amdgcn-- -mcpu=gfx1100 -O3 -mattr=+wavefrontsize32,-wavefrontsize64 -S < %s | FileCheck -check-prefixes=OPT,OPT-W32 %s
; RUN: opt -mtriple=amdgcn-- -mcpu=gfx1100 -O3 -mattr=-wavefrontsize32,+wavefrontsize64 -S < %s | FileCheck -check-prefixes=OPT,OPT-W64 %s

; GCN-LABEL: {{^}}fold_wavefrontsize:
; OPT-LABEL: define amdgpu_kernel void @fold_wavefrontsize(

; W32:       v_mov_b32_e32 [[V:v[0-9]+]], 32
; W64:       v_mov_b32_e32 [[V:v[0-9]+]], 64
; GCN:       store_{{dword|b32}} v{{.+}}, [[V]]

; OPT-W32:   store i32 32, i32 addrspace(1)* %arg, align 4
; OPT-W64:   store i32 64, i32 addrspace(1)* %arg, align 4
; OPT-WXX:   %tmp = tail call i32 @llvm.amdgcn.wavefrontsize()
; OPT-WXX:   store i32 %tmp, i32 addrspace(1)* %arg, align 4
; OPT-NEXT:  ret void

define amdgpu_kernel void @fold_wavefrontsize(i32 addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.wavefrontsize() #0
  store i32 %tmp, i32 addrspace(1)* %arg, align 4
  ret void
}

; GCN-LABEL: {{^}}fold_and_optimize_wavefrontsize:
; OPT-LABEL: define amdgpu_kernel void @fold_and_optimize_wavefrontsize(

; W32:       v_mov_b32_e32 [[V:v[0-9]+]], 1{{$}}
; W64:       v_mov_b32_e32 [[V:v[0-9]+]], 2{{$}}
; GCN-NOT:   cndmask
; GCN:       store_{{dword|b32}} v{{.+}}, [[V]]

; OPT-W32:   store i32 1, i32 addrspace(1)* %arg, align 4
; OPT-W64:   store i32 2, i32 addrspace(1)* %arg, align 4
; OPT-WXX:   %tmp = tail call i32 @llvm.amdgcn.wavefrontsize()
; OPT-WXX:   %tmp1 = icmp ugt i32 %tmp, 32
; OPT-WXX:   %tmp2 = select i1 %tmp1, i32 2, i32 1
; OPT-WXX:   store i32 %tmp2, i32 addrspace(1)* %arg
; OPT-NEXT:  ret void

define amdgpu_kernel void @fold_and_optimize_wavefrontsize(i32 addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.wavefrontsize() #0
  %tmp1 = icmp ugt i32 %tmp, 32
  %tmp2 = select i1 %tmp1, i32 2, i32 1
  store i32 %tmp2, i32 addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}fold_and_optimize_if_wavefrontsize:
; OPT-LABEL: define amdgpu_kernel void @fold_and_optimize_if_wavefrontsize(

; OPT:       bb:
; OPT-WXX:   %tmp = tail call i32 @llvm.amdgcn.wavefrontsize()
; OPT-WXX:   %tmp1 = icmp ugt i32 %tmp, 32
; OPT-WXX:   bb3:
; OPT-W64:   store i32 1, i32 addrspace(1)* %arg, align 4
; OPT-NEXT:  ret void

define amdgpu_kernel void @fold_and_optimize_if_wavefrontsize(i32 addrspace(1)* nocapture %arg) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.wavefrontsize() #0
  %tmp1 = icmp ugt i32 %tmp, 32
  br i1 %tmp1, label %bb2, label %bb3

bb2:                                              ; preds = %bb
  store i32 1, i32 addrspace(1)* %arg, align 4
  br label %bb3

bb3:                                              ; preds = %bb2, %bb
  ret void
}

declare i32 @llvm.amdgcn.wavefrontsize() #0

attributes #0 = { nounwind readnone speculatable }
