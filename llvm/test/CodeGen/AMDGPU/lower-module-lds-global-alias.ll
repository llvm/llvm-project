; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

;.
; @lds.1:  is aliased with @alias.to.lds.1, and @alias.to.lds.1 is used within kernel @k0.
;          Hence, @lds.1 is lowered.
; @lds.2:  is aliased with @alias.to.lds.2, and @alias.to.lds.2 is used within non-kernel @f0,
;          Hence, @lds.2 is lowered.
; @lds.3:  is used as initializer to global @gptr.3, and @gptr.3 is aliased with @alias.to.gptr.3,
;          and @alias.to.gptr.3 is used within kernel @k1. Hence, @lds.3 is lowered.
; @lds.4:  is used as initializer to global @gptr.4, and @gptr.4 is aliased with @alias.to.gptr.4,
;          and @alias.to.gptr.4 is used within non-kernel @f1. Hence, @lds.4 is lowered.
; @lds.5:  is aliased with @alias.to.lds.5, but neither @lds.5 nor @alias.to.lds.5 is used anywhere.
;          Hence, @lds.5 is not lowered.
; @lds.6:  is used as initializer to global @gptr.6, and @gptr.6 is aliased with @alias.to.gptr.6.
;          But none of them are used anywhere. Hence, @lds.6 is not lowered.
;.

; CHECK: %llvm.amdgcn.module.lds.t = type { [4 x i8], [3 x i8], [1 x i8], [2 x i8] }

; CHECK-NOT: @lds.1
; CHECK-NOT: @lds.2
; CHECK-NOT: @lds.3
; CHECK-NOT: @lds.4
; CHECK: @lds.5 = internal unnamed_addr addrspace(3) global [5 x i8] undef, align 8
; CHECK: @lds.6 = internal unnamed_addr addrspace(3) global [6 x i8] undef, align 8
@lds.1 = internal unnamed_addr addrspace(3) global [1 x i8] undef, align 1
@lds.2 = internal unnamed_addr addrspace(3) global [2 x i8] undef, align 2
@lds.3 = internal unnamed_addr addrspace(3) global [3 x i8] undef, align 4
@lds.4 = internal unnamed_addr addrspace(3) global [4 x i8] undef, align 4
@lds.5 = internal unnamed_addr addrspace(3) global [5 x i8] undef, align 8
@lds.6 = internal unnamed_addr addrspace(3) global [6 x i8] undef, align 8

; CHECK: @gptr.3 = addrspace(1) global i64* addrspacecast (i64 addrspace(3)* bitcast ([3 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 1) to i64 addrspace(3)*) to i64*), align 8
; CHECK: @gptr.4 = addrspace(1) global i64* addrspacecast (i64 addrspace(3)* bitcast (%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds to i64 addrspace(3)*) to i64*), align 8
; CHECK: @gptr.6 = addrspace(1) global i64* addrspacecast (i64 addrspace(3)* bitcast ([6 x i8] addrspace(3)* @lds.6 to i64 addrspace(3)*) to i64*), align 8
@gptr.3 = addrspace(1) global i64* addrspacecast ([3 x i8] addrspace(3)* @lds.3 to i64*), align 8
@gptr.4 = addrspace(1) global i64* addrspacecast ([4 x i8] addrspace(3)* @lds.4 to i64*), align 8
@gptr.6 = addrspace(1) global i64* addrspacecast ([6 x i8] addrspace(3)* @lds.6 to i64*), align 8

; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t undef, align 4
; CHECK: @llvm.compiler.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 0, i32 0) to i8*)], section "llvm.metadata"

; CHECK: @alias.to.lds.1 = alias [1 x i8], getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 2)
; CHECK: @alias.to.lds.2 = alias [2 x i8], getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 3)
; CHECK: @alias.to.gptr.3 = alias i64*, i64* addrspace(1)* @gptr.3
; CHECK: @alias.to.gptr.4 = alias i64*, i64* addrspace(1)* @gptr.4
; CHECK: @alias.to.lds.5 = alias [5 x i8], [5 x i8] addrspace(3)* @lds.5
; CHECK: @alias.to.gptr.6 = alias i64*, i64* addrspace(1)* @gptr.6
@alias.to.lds.1 = alias [1 x i8], [1 x i8] addrspace(3)* @lds.1
@alias.to.lds.2 = alias [2 x i8], [2 x i8] addrspace(3)* @lds.2
@alias.to.gptr.3 = alias i64*, i64* addrspace(1)* @gptr.3
@alias.to.gptr.4 = alias i64*, i64* addrspace(1)* @gptr.4
@alias.to.lds.5 = alias [5 x i8], [5 x i8] addrspace(3)* @lds.5
@alias.to.gptr.6 = alias i64*, i64* addrspace(1)* @gptr.6

; CHECK-LABEL: @f1
; CHECK:   %ld = load i64*, i64* addrspace(1)* @alias.to.gptr.4, align 8
; CHECK:   ret void
define void @f1() {
  %ld = load i64*, i64* addrspace(1)* @alias.to.gptr.4
  ret void
}

; CHECK-LABEL: @f0
; CHECK:   %bc = bitcast [2 x i8] addrspace(3)* @alias.to.lds.2 to i8 addrspace(3)*
; CHECK:   store i8 1, i8 addrspace(3)* %bc, align 2
; CHECK:   ret void
define void @f0() {
  %bc = bitcast [2 x i8] addrspace(3)* @alias.to.lds.2 to i8 addrspace(3)*
  store i8 1, i8 addrspace(3)* %bc, align 2
  ret void
}

; CHECK-LABEL: @k1
; CHECK-LABEL:   call void @llvm.donothing() [ "ExplicitUse"(%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds) ]
; CHECK-LABEL:   %ld = load i64*, i64* addrspace(1)* @alias.to.gptr.3, align 8
; CHECK-LABEL:   ret void
define amdgpu_kernel void @k1() {
  %ld = load i64*, i64* addrspace(1)* @alias.to.gptr.3
  ret void
}

; CHECK-LABEL: @k0
; CHECK-LABEL:   call void @llvm.donothing() [ "ExplicitUse"(%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds) ]
; CHECK-LABEL:   %bc = bitcast [1 x i8] addrspace(3)* @alias.to.lds.1 to i8 addrspace(3)*
; CHECK-LABEL:   store i8 1, i8 addrspace(3)* %bc, align 1
; CHECK-LABEL:   ret void
define amdgpu_kernel void @k0() {
  %bc = bitcast [1 x i8] addrspace(3)* @alias.to.lds.1 to i8 addrspace(3)*
  store i8 1, i8 addrspace(3)* %bc, align 1
  ret void
}
