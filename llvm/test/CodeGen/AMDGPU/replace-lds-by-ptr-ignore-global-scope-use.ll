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

; CHECK: @global_var = addrspace(1) global float* addrspacecast (float addrspace(3)* bitcast ([4 x i32] addrspace(3)* @lds to float addrspace(3)*) to float*), align 8
; CHECK: @llvm.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (i16 addrspace(3)* @lds.1 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"
; CHECK: @llvm.compiler.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (i32 addrspace(3)* @lds.2 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"
; CHECK: @alias.to.lds.3 = alias [1 x i8], [1 x i8] addrspace(3)* @lds.3
@global_var = addrspace(1) global float* addrspacecast ([4 x i32] addrspace(3)* @lds to float*), align 8
@llvm.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (i16 addrspace(3)* @lds.1 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (i32 addrspace(3)* @lds.2 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"
@alias.to.lds.3 = alias [1 x i8], [1 x i8] addrspace(3)* @lds.3

; CHECK-NOT: @lds.ptr
; CHECK-NOT: @lds.1.ptr
; CHECK-NOT: @lds.2.ptr
; CHECK-NOT: @lds.3.ptr

define void @f0() {
; CHECK-LABEL: entry:
; CHECK:   %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @lds, i32 0, i32 0
; CHECK:   %ld1 = load i16, i16 addrspace(3)* @lds.1
; CHECK:   %ld2 = load i32, i32 addrspace(3)* @lds.2
; CHECK:   %gep2 = getelementptr inbounds [1 x i8], [1 x i8] addrspace(3)* @lds.3, i32 0, i32 0
; CHECK:   ret void
entry:
  %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @lds, i32 0, i32 0
  %ld1 = load i16, i16 addrspace(3)* @lds.1
  %ld2 = load i32, i32 addrspace(3)* @lds.2
  %gep2 = getelementptr inbounds [1 x i8], [1 x i8] addrspace(3)* @lds.3, i32 0, i32 0
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
