; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -passes=load-store-vectorizer -S -o - %s | FileCheck -check-prefixes=GCN %s
; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -passes=load-store-vectorizer -S -o - %s | FileCheck -check-prefixes=GCN %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

; Checks that there is no crash when there are multiple tails
; for a the same head starting a chain.
@0 = internal addrspace(3) global [16384 x i32] undef

; GCN-LABEL: @no_crash(
; GCN: store <2 x i32> zeroinitializer
; GCN: store i32 0
; GCN: store i32 0

define amdgpu_kernel void @no_crash(i32 %arg) {
  %tmp2 = add i32 %arg, 14
  %tmp3 = getelementptr [16384 x i32], ptr addrspace(3) @0, i32 0, i32 %tmp2
  %tmp4 = add i32 %arg, 15
  %tmp5 = getelementptr [16384 x i32], ptr addrspace(3) @0, i32 0, i32 %tmp4

  store i32 0, ptr addrspace(3) %tmp3, align 4
  store i32 0, ptr addrspace(3) %tmp5, align 4
  store i32 0, ptr addrspace(3) %tmp5, align 4
  store i32 0, ptr addrspace(3) %tmp5, align 4

  ret void
}

; Check adjacent memory locations are properly matched and the
; longest chain vectorized

; GCN-LABEL: @interleave_get_longest

; GCN: load <2 x i32>{{.*}} %tmp1
; GCN: store <2 x i32> zeroinitializer{{.*}} %tmp1
; GCN: load <2 x i32>{{.*}} %tmp2
; GCN: load <2 x i32>{{.*}} %tmp4
; GCN: load i32{{.*}} %tmp5
; GCN: load i32{{.*}} %tmp5

define amdgpu_kernel void @interleave_get_longest(i32 %arg) {
  %a1 = add i32 %arg, 1
  %a2 = add i32 %arg, 2
  %a3 = add i32 %arg, 3
  %a4 = add i32 %arg, 4
  %tmp1 = getelementptr [16384 x i32], ptr addrspace(3) @0, i32 0, i32 %arg
  %tmp2 = getelementptr [16384 x i32], ptr addrspace(3) @0, i32 0, i32 %a1
  %tmp3 = getelementptr [16384 x i32], ptr addrspace(3) @0, i32 0, i32 %a2
  %tmp4 = getelementptr [16384 x i32], ptr addrspace(3) @0, i32 0, i32 %a3
  %tmp5 = getelementptr [16384 x i32], ptr addrspace(3) @0, i32 0, i32 %a4

  %l1 = load i32, ptr addrspace(3) %tmp2, align 4
  %l2 = load i32, ptr addrspace(3) %tmp1, align 4
  store i32 0, ptr addrspace(3) %tmp2, align 4
  store i32 0, ptr addrspace(3) %tmp1, align 4
  %l3 = load i32, ptr addrspace(3) %tmp2, align 4
  %l4 = load i32, ptr addrspace(3) %tmp3, align 4
  %l5 = load i32, ptr addrspace(3) %tmp4, align 4
  %l6 = load i32, ptr addrspace(3) %tmp5, align 4
  %l7 = load i32, ptr addrspace(3) %tmp5, align 4
  %l8 = load i32, ptr addrspace(3) %tmp5, align 4

  ret void
}

