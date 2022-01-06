; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

;.
; Kernel LDS lowering.
;.
; @lds.1:  is part of @llvm.used list, and also it is used within kernel, hence it is lowered.
; @lds.2:  is part of @llvm.compiler.used list, and also it is used within kernel, hence it is lowered.
; @lds.3:  is used as initializer to @gptr.3, hence @lds.3 is not lowered, though it is used within kernel.
; @lds.4:  is used as initializer to @gptr.4, hence @lds.4 is not lowered, though it is used within kernel,
;          irrespective of the uses of @gptr.4 itself ( @gptr.4 is part of llvm.compiler.used list ).
; @lds.5:  is part of @llvm.used list, but is not used within kernel, hence it is not lowered.
; @lds.6:  is part of @llvm.compiler.used list, but is not used within kernel, hence it is not lowered.
;.

; CHECK: %llvm.amdgcn.kernel.k0.lds.t = type { i32, i16 }

; CHECK-NOT: @lds.1
; CHECK-NOT: @lds.2
; CHECK: @lds.3 = addrspace(3) global i64 undef, align 8
; CHECK: @lds.4 = addrspace(3) global float undef, align 4
; CHECK: @lds.5 = addrspace(3) global i16 undef, align 2
; CHECK: @lds.6 = addrspace(3) global i32 undef, align 4
@lds.1 = addrspace(3) global i16 undef, align 2
@lds.2 = addrspace(3) global i32 undef, align 4
@lds.3 = addrspace(3) global i64 undef, align 8
@lds.4 = addrspace(3) global float undef, align 4
@lds.5 = addrspace(3) global i16 undef, align 2
@lds.6 = addrspace(3) global i32 undef, align 4

; CHECK: @gptr.3 = addrspace(1) global i64* addrspacecast (i64 addrspace(3)* @lds.3 to i64*), align 8
; CHECK: @gptr.4 = addrspace(1) global i64* addrspacecast (i64 addrspace(3)* bitcast (float addrspace(3)* @lds.4 to i64 addrspace(3)*) to i64*), align 8
@gptr.3 = addrspace(1) global i64* addrspacecast (i64 addrspace(3)* @lds.3 to i64*), align 8
@gptr.4 = addrspace(1) global i64* addrspacecast (float addrspace(3)* @lds.4 to i64*), align 8

; CHECK: @llvm.amdgcn.kernel.k0.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k0.lds.t undef, align 4

; CHECK: @llvm.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (i16 addrspace(3)* @lds.5 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"
; CHECK: @llvm.compiler.used = appending global [2 x i8*] [i8* addrspacecast (i8 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.4 to i8 addrspace(1)*) to i8*), i8* addrspacecast (i8 addrspace(3)* bitcast (i32 addrspace(3)* @lds.6 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"
@llvm.used = appending global [2 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (i16 addrspace(3)* @lds.1 to i8 addrspace(3)*) to i8*), i8* addrspacecast (i8 addrspace(3)* bitcast (i16 addrspace(3)* @lds.5 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"
@llvm.compiler.used = appending global [3 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (i32 addrspace(3)* @lds.2 to i8 addrspace(3)*) to i8*), i8* addrspacecast (i8 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.4 to i8 addrspace(1)*) to i8*), i8* addrspacecast (i8 addrspace(3)* bitcast (i32 addrspace(3)* @lds.6 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"

; CHECK-LABEL: @k0()
; CHECK:   %ld.lds.1 = load i16, i16 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, %llvm.amdgcn.kernel.k0.lds.t addrspace(3)* @llvm.amdgcn.kernel.k0.lds, i32 0, i32 1), align 4
; CHECK:   %ld.lds.2 = load i32, i32 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, %llvm.amdgcn.kernel.k0.lds.t addrspace(3)* @llvm.amdgcn.kernel.k0.lds, i32 0, i32 0), align 4
; CHECK:   %ld.lds.3 = load i64, i64 addrspace(3)* @lds.3, align 4
; CHECK:   %ld.lds.4 = load float, float addrspace(3)* @lds.4, align 4
; CHECK:   ret void
define amdgpu_kernel void @k0() {
  %ld.lds.1 = load i16, i16 addrspace(3)* @lds.1
  %ld.lds.2 = load i32, i32 addrspace(3)* @lds.2
  %ld.lds.3 = load i64, i64 addrspace(3)* @lds.3
  %ld.lds.4 = load float, float addrspace(3)* @lds.4
  ret void
}
