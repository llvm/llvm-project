; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

; Variables that are not lowered by this pass are left unchanged
; CHECK-NOT: asm
; CHECK-NOT: llvm.amdgcn.module.lds
; CHECK-NOT: llvm.amdgcn.module.lds.t

; var1, var2 would be transformed were they used from a non-kernel function
; CHECK-NOT: @var1 =
; CHECK: @var2 = addrspace(3) global float undef
@var1 = addrspace(3) global i32 undef
@var2 = addrspace(3) global float undef

; constant variables are left to the optimizer / error diagnostics
; CHECK: @const_undef = addrspace(3) constant i32 undef
; CHECK: @const_with_init = addrspace(3) constant i64 8
@const_undef = addrspace(3) constant i32 undef
@const_with_init = addrspace(3) constant i64 8

; External and constant are both left to the optimizer / error diagnostics
; CHECK: @extern = external addrspace(3) global i32
@extern = external addrspace(3) global i32

; Use of an addrspace(3) variable with an initializer is skipped,
; so as to preserve the unimplemented error from llc
; CHECK: @with_init = addrspace(3) global i64 0
@with_init = addrspace(3) global i64 0

; Only local addrspace variables are transformed
; CHECK: @addr4 = addrspace(4) global i64 undef
@addr4 = addrspace(4) global i64 undef

; Assign to self is treated as any other initializer, i.e. ignored by this pass
; CHECK: @toself = addrspace(3) global float addrspace(3)* bitcast (float addrspace(3)* addrspace(3)* @toself to float addrspace(3)*), align 8
@toself = addrspace(3) global float addrspace(3)* bitcast (float addrspace(3)* addrspace(3)* @toself to float addrspace(3)*), align 8

; Use by .used lists doesn't trigger lowering
; CHECK-NOT: @llvm.used =
@llvm.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (i32 addrspace(3)* @var1 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"

; CHECK: @llvm.compiler.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (float addrspace(3)* @var2 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (float addrspace(3)* @var2 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"

; Access from a function would cause lowering for non-excluded cases
; CHECK-LABEL: @use_variables()
; CHECK: %c0 = load i32, i32 addrspace(3)* @const_undef, align 4
; CHECK: %c1 = load i64, i64 addrspace(3)* @const_with_init, align 4
; CHECK: %v0 = atomicrmw add i64 addrspace(3)* @with_init, i64 1 seq_cst
; CHECK: %v1 = cmpxchg i32 addrspace(3)* @extern, i32 4, i32 %c0 acq_rel monotonic
; CHECK: %v2 = atomicrmw add i64 addrspace(4)* @addr4, i64 %c1 monotonic
define void @use_variables() {
  %c0 = load i32, i32 addrspace(3)* @const_undef, align 4
  %c1 = load i64, i64 addrspace(3)* @const_with_init, align 4
  %v0 = atomicrmw add i64 addrspace(3)* @with_init, i64 1 seq_cst
  %v1 = cmpxchg i32 addrspace(3)* @extern, i32 4, i32 %c0 acq_rel monotonic
  %v2 = atomicrmw add i64 addrspace(4)* @addr4, i64 %c1 monotonic
  ret void
}

; CHECK-LABEL: @kern_use()
; CHECK: %inc = atomicrmw add i32 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.kern_use.lds.t, %llvm.amdgcn.kernel.kern_use.lds.t addrspace(3)* @llvm.amdgcn.kernel.kern_use.lds, i32 0, i32 0), i32 1 monotonic, align 4
define amdgpu_kernel void @kern_use() {
  %inc = atomicrmw add i32 addrspace(3)* @var1, i32 1 monotonic
  call void @use_variables()
  ret void
}
