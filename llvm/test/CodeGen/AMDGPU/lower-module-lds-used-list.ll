; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s

; Check new struct is added to compiler.used and that the replaced variable is removed

; CHECK: %llvm.amdgcn.module.lds.t = type { float }
; CHECK: @ignored = addrspace(1) global i64 0

; @ignored still in list, @tolower removed, llvm.amdgcn.module.lds appended
; Start with one value to replace and one to ignore in the .use list

; @ignored still in list, @tolower removed
; CHECK: @llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @ignored to ptr)], section "llvm.metadata"

; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t poison, align 8

; CHECK-NOT: @tolower

@tolower = addrspace(3) global float poison, align 8

; A variable that is unchanged by pass
@ignored = addrspace(1) global i64 0


@llvm.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(3) @tolower to ptr), ptr addrspacecast (ptr addrspace(1) @ignored to ptr)], section "llvm.metadata"

; @ignored still in list, @tolower removed, llvm.amdgcn.module.lds appended
; CHECK: @llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(1) @ignored to ptr), ptr addrspacecast (ptr addrspace(3) @llvm.amdgcn.module.lds to ptr)], section "llvm.metadata"

@llvm.compiler.used = appending global [2 x ptr] [ptr addrspacecast (ptr addrspace(3) @tolower to ptr), ptr addrspacecast (ptr addrspace(1) @ignored to ptr)], section "llvm.metadata"


; Functions that are not called are ignored by the lowering
define amdgpu_kernel void @call_func() {
  call void @func()
  ret void
}

; CHECK-LABEL: @func()
; CHECK: %dec = atomicrmw fsub ptr addrspace(3) @llvm.amdgcn.module.lds, float 1.000000e+00 monotonic, align 8
define void @func() {
  %dec = atomicrmw fsub ptr addrspace(3) @tolower, float 1.0 monotonic
  %unused0 = atomicrmw add ptr addrspace(1) @ignored, i64 1 monotonic
  ret void
}
