; Test that we don't crash.
; RUN: opt < %s -passes=alignment-from-assumptions -S

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @vectorize_global_local(ptr addrspace(1) nocapture readonly %arg, ptr addrspace(3) nocapture %arg1) {
bb:
  %tmp2 = getelementptr inbounds i32, ptr addrspace(1) %arg, i64 1
  call void @llvm.assume(i1 true) [ "align"(ptr addrspace(1) %tmp2, i64 4) ]
  %tmp3 = load i32, ptr addrspace(1) %tmp2, align 4
  %tmp4 = getelementptr inbounds i32, ptr addrspace(3) %arg1, i32 1
  store i32 %tmp3, ptr addrspace(3) %tmp4, align 4
  ret void
}
declare void @llvm.assume(i1 noundef)
