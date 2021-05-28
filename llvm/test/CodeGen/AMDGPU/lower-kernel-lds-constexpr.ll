; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

; CHECK: %llvm.amdgcn.kernel.k2.lds.t = type { i32 }
; CHECK-NOT: %llvm.amdgcn.kernel.k4.lds.t

@lds.1 = internal unnamed_addr addrspace(3) global [2 x i8] undef, align 1

; Use constant from different kernels
;.
; CHECK: @lds.1 = internal unnamed_addr addrspace(3) global [2 x i8] undef, align 1
; CHECK: @llvm.amdgcn.kernel.k2.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k2.lds.t undef, align 4
;.
define amdgpu_kernel void @k0(i64 %x) {
; CHECK-LABEL: @k0(
; CHECK-NEXT:    %ptr = getelementptr inbounds i8, i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(3)* @lds.1, i32 0, i32 0) to i8*), i64 %x
; CHECK-NEXT:    store i8 1, i8* %ptr, align 1
; CHECK-NEXT:    ret void
;
  %ptr = getelementptr inbounds i8, i8* addrspacecast ([2 x i8] addrspace(3)* @lds.1 to i8*), i64 %x
  store i8 1, i8 addrspace(0)* %ptr, align 1
  ret void
}

define amdgpu_kernel void @k1(i64 %x) {
; CHECK-LABEL: @k1(
; CHECK-NEXT:    %ptr = getelementptr inbounds i8, i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(3)* @lds.1, i32 0, i32 0) to i8*), i64 %x
; CHECK-NEXT:    store i8 1, i8* %ptr, align 1
; CHECK-NEXT:    ret void
;
  %ptr = getelementptr inbounds i8, i8* addrspacecast ([2 x i8] addrspace(3)* @lds.1 to i8*), i64 %x
  store i8 1, i8 addrspace(0)* %ptr, align 1
  ret void
}

@lds.2 = internal unnamed_addr addrspace(3) global i32 undef, align 4

; Use constant twice from the same kernel
define amdgpu_kernel void @k2(i64 %x) {
; CHECK-LABEL: @k2(
; CHECK-NEXT:    %ptr1 = bitcast i32 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k2.lds.t, %llvm.amdgcn.kernel.k2.lds.t addrspace(3)* @llvm.amdgcn.kernel.k2.lds, i32 0, i32 0) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 1, i8 addrspace(3)* %ptr1, align 4
; CHECK-NEXT:    %ptr2 = bitcast i32 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k2.lds.t, %llvm.amdgcn.kernel.k2.lds.t addrspace(3)* @llvm.amdgcn.kernel.k2.lds, i32 0, i32 0) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 2, i8 addrspace(3)* %ptr2, align 4
; CHECK-NEXT:    ret void
;
  %ptr1 = bitcast i32 addrspace(3)* @lds.2 to i8 addrspace(3)*
  store i8 1, i8 addrspace(3)* %ptr1, align 4
  %ptr2 = bitcast i32 addrspace(3)* @lds.2 to i8 addrspace(3)*
  store i8 2, i8 addrspace(3)* %ptr2, align 4
  ret void
}

@lds.3 = internal unnamed_addr addrspace(3) global [32 x i8] undef, align 1

; Use constant twice from the same kernel but a different other constant.
define amdgpu_kernel void @k3(i64 %x) {
; CHECK-LABEL: @k3(
; CHECK-NEXT:    %ptr1 = addrspacecast i64 addrspace(3)* bitcast (i8 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k3.lds.t, %llvm.amdgcn.kernel.k3.lds.t addrspace(3)* @llvm.amdgcn.kernel.k3.lds, i32 0, i32 0, i32 16) to i64 addrspace(3)*) to i64*
; CHECK-NEXT:    store i64 1, i64* %ptr1, align 1
; CHECK-NEXT:    %ptr2 = addrspacecast i64 addrspace(3)* bitcast (i8 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k3.lds.t, %llvm.amdgcn.kernel.k3.lds.t addrspace(3)* @llvm.amdgcn.kernel.k3.lds, i32 0, i32 0, i32 24) to i64 addrspace(3)*) to i64*
; CHECK-NEXT:    store i64 2, i64* %ptr2, align 1
;
  %ptr1 = addrspacecast i64 addrspace(3)* bitcast (i8 addrspace(3)* getelementptr inbounds ([32 x i8], [32 x i8] addrspace(3)* @lds.3, i32 0, i32 16) to i64 addrspace(3)*) to i64*
  store i64 1, i64* %ptr1, align 1
  %ptr2 = addrspacecast i64 addrspace(3)* bitcast (i8 addrspace(3)* getelementptr inbounds ([32 x i8], [32 x i8] addrspace(3)* @lds.3, i32 0, i32 24) to i64 addrspace(3)*) to i64*
  store i64 2, i64* %ptr2, align 1
  ret void
}

; @lds.1 is used from constant expressions in different kernels.
; Make sure we do not create a structure for it as we cannot handle it yet.
define amdgpu_kernel void @k4(i64 %x) {
; CHECK-LABEL: @k4(
; CHECK-NEXT:    %ptr = getelementptr inbounds i8, i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(3)* @lds.1, i32 0, i32 0) to i8*), i64 %x
; CHECK-NEXT:    store i8 1, i8* %ptr, align 1
; CHECK-NEXT:    ret void
;
  %ptr = getelementptr inbounds i8, i8* addrspacecast ([2 x i8] addrspace(3)* @lds.1 to i8*), i64 %x
  store i8 1, i8 addrspace(0)* %ptr, align 1
  ret void
}
