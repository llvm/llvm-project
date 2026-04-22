; RUN: split-file %s %t

; RUN: not --crash llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -o - %t/null-named-barrier-kernel.ll 2>&1   | FileCheck %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -o - %t/null-named-barrier-kernel.ll 2>&1                | FileCheck %s

; RUN: not --crash llc -global-isel -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250  -o - %t/null-named-barrier-func.ll 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -o - %t/null-named-barrier-func.ll 2>&1               | FileCheck %s

; CHECK: named barrier GV cannot be used to represent the NULL named barrier

;--- null-named-barrier-kernel.ll

@bar = internal addrspace(15) global [2 x target("amdgcn.named.barrier", 0)] poison, !absolute_symbol !0

define amdgpu_kernel void @func1() {
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) @bar)
    ret void
}

!0 = !{ i32 0, i32 1 }

;--- null-named-barrier-func.ll

@bar = internal addrspace(15) global [2 x target("amdgcn.named.barrier", 0)] poison, !absolute_symbol !0

define void @func1() {
    call void @llvm.amdgcn.s.barrier.join(ptr addrspace(15) @bar)
    ret void
}

!0 = !{ i32 0, i32 1 }
