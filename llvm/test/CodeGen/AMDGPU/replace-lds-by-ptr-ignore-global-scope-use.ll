; RUN: opt -S -mtriple=amdgcn--  -amdgpu-replace-lds-use-with-pointer -amdgpu-enable-lds-replace-with-pointer=true < %s | FileCheck %s

; DESCRIPTION:
;
; None of lds are pointer-replaced since they are all used in global scope in one or the other way.
;

; CHECK: @lds = internal addrspace(3) global [4 x i32] undef, align 4
; CHECK: @lds.1 = addrspace(3) global i16 undef, align 2
; CHECK: @lds.2 = addrspace(3) global i32 undef, align 4
; CHECK: @lds.3 = internal unnamed_addr addrspace(3) global [1 x i8] undef, align 1
@lds = internal addrspace(3) global [4 x i32] undef, align 4
@lds.1 = addrspace(3) global i16 undef, align 2
@lds.2 = addrspace(3) global i32 undef, align 4
@lds.3 = internal unnamed_addr addrspace(3) global [1 x i8] undef, align 1

; CHECK: @global_var = addrspace(1) global ptr addrspacecast (ptr addrspace(3) @lds to ptr), align 8
; CHECK: @llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @lds.1 to ptr)], section "llvm.metadata"
; CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @lds.2 to ptr)], section "llvm.metadata"
; CHECK: @alias.to.lds.3 = alias [1 x i8], ptr addrspace(3) @lds.3
@global_var = addrspace(1) global ptr addrspacecast (ptr addrspace(3) @lds to ptr), align 8
@llvm.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @lds.1 to ptr)], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @lds.2 to ptr)], section "llvm.metadata"
@alias.to.lds.3 = alias [1 x i8], ptr addrspace(3) @lds.3

; CHECK-NOT: @lds.ptr
; CHECK-NOT: @lds.1.ptr
; CHECK-NOT: @lds.2.ptr
; CHECK-NOT: @lds.3.ptr

define void @f0() {
; CHECK-LABEL: entry:
; CHECK:   %ld1 = load i16, ptr addrspace(3) @lds.1
; CHECK:   %ld2 = load i32, ptr addrspace(3) @lds.2
; CHECK:   ret void
entry:
  %ld1 = load i16, ptr addrspace(3) @lds.1
  %ld2 = load i32, ptr addrspace(3) @lds.2
  ret void
}

define protected amdgpu_kernel void @k0() {
; CHECK-LABEL: entry:
; CHECK:   call void @f0()
; CHECK:   ret void
entry:
  call void @f0()
  ret void
}
