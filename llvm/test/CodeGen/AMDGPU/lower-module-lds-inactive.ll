; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

; Variables that are not lowered by this pass are left unchanged
; CHECK-NOT: asm
; CHECK-NOT: llvm.amdgcn.module.lds
; CHECK-NOT: llvm.amdgcn.module.lds.t

; var1 is removed, var2 stays because it's in compiler.used
; CHECK-NOT: @var1
; CHECK: @var2 = addrspace(3) global float undef
@var1 = addrspace(3) global i32 undef
@var2 = addrspace(3) global float undef

; constant variables are left to the optimizer / error diagnostics
; CHECK: @const_undef = addrspace(3) constant i32 undef
; CHECK: @const_with_init = addrspace(3) constant i64 8
@const_undef = addrspace(3) constant i32 undef
@const_with_init = addrspace(3) constant i64 8

; Use of an addrspace(3) variable with an initializer is skipped,
; so as to preserve the unimplemented error from llc
; CHECK: @with_init = addrspace(3) global i64 0
@with_init = addrspace(3) global i64 0

; Only local addrspace variables are transformed
; CHECK: @addr4 = addrspace(4) global i64 undef
@addr4 = addrspace(4) global i64 undef

; Assign to self is treated as any other initializer, i.e. ignored by this pass
; CHECK: @toself = addrspace(3) global ptr addrspace(3) @toself, align 8
@toself = addrspace(3) global ptr addrspace(3) @toself, align 8

; Use by .used lists doesn't trigger lowering
; CHECK-NOT: @llvm.used =
@llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @var1 to ptr)], section "llvm.metadata"

; CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @var2 to ptr)], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @var2 to ptr)], section "llvm.metadata"

; Access from a function would cause lowering for non-excluded cases
; CHECK-LABEL: @use_variables()
; CHECK: %c0 = load i32, ptr addrspace(3) @const_undef, align 4
; CHECK: %c1 = load i64, ptr addrspace(3) @const_with_init, align 4
; CHECK: %v0 = atomicrmw add ptr addrspace(3) @with_init, i64 1 seq_cst
; CHECK: %v1 = atomicrmw add ptr addrspace(4) @addr4, i64 %c1 monotonic
define void @use_variables() {
  %c0 = load i32, ptr addrspace(3) @const_undef, align 4
  %c1 = load i64, ptr addrspace(3) @const_with_init, align 4
  %v0 = atomicrmw add ptr addrspace(3) @with_init, i64 1 seq_cst
  %v1 = atomicrmw add ptr addrspace(4) @addr4, i64 %c1 monotonic
  ret void
}

; CHECK-LABEL: @kern_use()
; CHECK: %inc = atomicrmw add ptr addrspace(3) @llvm.amdgcn.kernel.kern_use.lds, i32 1 monotonic, align 4
define amdgpu_kernel void @kern_use() {
  %inc = atomicrmw add ptr addrspace(3) @var1, i32 1 monotonic
  call void @use_variables()
  ret void
}
