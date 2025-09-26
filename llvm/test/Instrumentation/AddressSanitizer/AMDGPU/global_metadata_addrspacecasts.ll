; RUN: opt < %s -passes=asan -S | FileCheck %s
target triple = "amdgcn-amd-amdhsa"

@g = addrspace(1) global [1 x i32] zeroinitializer, align 4

;CHECK: llvm.asan.globals

!llvm.asan.globals = !{!0, !1}
!0 = !{ptr addrspace(1) @g, null, !"name", i1 false, i1 false}
!1 = !{ptr addrspacecast (ptr addrspace(1) @g to  ptr), null, !"name", i1 false, i1 false}
