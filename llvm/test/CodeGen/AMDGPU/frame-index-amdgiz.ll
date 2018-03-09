; RUN: llc -mtriple=amdgcn---amdgiz -mcpu=kaveri -verify-machineinstrs < %s | FileCheck %s
;
; The original OpenCL kernel:
; kernel void f(global int *a, int i,  int j) {
;  int x[100];
;  x[i] = 7;
;  a[0] = x[j];
; }
; clang -cc1 -triple amdgcn---amdgizcl -emit-llvm -o -

target datalayout = "e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5"

define amdgpu_kernel void @f(i32 addrspace(1)* nocapture %a, i32 %i, i32 %j) local_unnamed_addr #0 {
entry:
; CHECK: s_load_dwordx2 s[4:5], s[0:1], 0x9
; CHECK: s_load_dword s2, s[0:1], 0xb
; CHECK: s_load_dword s0, s[0:1], 0xc
; CHECK: s_mov_b32 s8, SCRATCH_RSRC_DWORD0
; CHECK: s_mov_b32 s9, SCRATCH_RSRC_DWORD1
; CHECK: s_mov_b32 s10, -1
; CHECK: s_waitcnt lgkmcnt(0)
; CHECK: s_lshl_b32 s1, s2, 2
; CHECK: v_mov_b32_e32 v0, 4
; CHECK: s_mov_b32 s11, 0xe8f000
; CHECK: v_add_i32_e32 v1, vcc, s1, v0
; CHECK: v_mov_b32_e32 v2, 7
; CHECK: s_lshl_b32 s0, s0, 2
; CHECK: buffer_store_dword v2, v1, s[8:11], s3 offen
; CHECK: v_add_i32_e32 v0, vcc, s0, v0
; CHECK: s_mov_b32 s7, 0xf000
; CHECK: s_mov_b32 s6, -1
; CHECK: buffer_load_dword v0, v0, s[8:11], s3 offen
; CHECK: s_waitcnt vmcnt(0)
; CHECK: buffer_store_dword v0, off, s[4:7], 0
; CHECK: s_endpgm

  %x = alloca [100 x i32], align 4, addrspace(5)
  %0 = bitcast [100 x i32] addrspace(5)* %x to i8 addrspace(5)*
  call void @llvm.lifetime.start.p5i8(i64 400, i8 addrspace(5)* nonnull %0) #0
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32] addrspace(5)* %x, i32 0, i32 %i
  store i32 7, i32 addrspace(5)* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [100 x i32], [100 x i32] addrspace(5)* %x, i32 0, i32 %j
  %1 = load i32, i32 addrspace(5)* %arrayidx2, align 4
  store i32 %1, i32 addrspace(1)* %a, align 4
  call void @llvm.lifetime.end.p5i8(i64 400, i8 addrspace(5)* nonnull %0) #0
  ret void
}

declare void @llvm.lifetime.start.p5i8(i64, i8 addrspace(5)* nocapture) #1

declare void @llvm.lifetime.end.p5i8(i64, i8 addrspace(5)* nocapture) #1

attributes #0 = { nounwind }
attributes #1 = { argmemonly nounwind }
