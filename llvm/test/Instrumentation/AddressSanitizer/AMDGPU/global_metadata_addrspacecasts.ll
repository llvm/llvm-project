; RUN: opt < %s -passes=asan -S | FileCheck %s
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

@g = addrspace(1) global [1 x i32] zeroinitializer, align 4

;CHECK: llvm.asan.globals

!llvm.asan.globals = !{!0, !1}
!0 = !{ptr addrspace(1) @g, null, !"name", i1 false, i1 false}
!1 = !{ptr addrspacecast (ptr addrspace(1) @g to  ptr), null, !"name", i1 false, i1 false}
