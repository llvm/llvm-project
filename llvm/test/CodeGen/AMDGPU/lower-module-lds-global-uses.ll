; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

;.
; @lds.1:  is part of @llvm.used list, and is no-where used. Hence it is not lowered.
; @lds.2:  is part of @llvm.compiler.used list, and is no-where used. Hence it is not lowered.
; @lds.3:  is used as initializer to @gptr.3, and is no-where used. @gptr.3 itself is also not
;          used anywhere else, hence @lds.3 is not lowered.
; @lds.4:  is used as initializer to @gptr.4, and is no-where used. @gptr.4 is part of
;          @llvm.compiler.used list, but is no-where else used. hence @lds.4 is not lowered.
;
; @lds.5:  is used as initializer to @gptr.5, and is no-where used. @gptr.5 is part of
;          @llvm.compiler.used list, but is also used within kernel @k0. Hence @lds.5 is lowered.
; @lds.6:  is used as initializer to @gptr.6, and is no-where used. @gptr.6 is part of
;          @llvm.compiler.used list, but is also used within non-kernel function @f0. Hence @lds.6 is lowered.
; @lds.7:  is used as initializer to @gptr.7, and is no-where used. @gptr.7 is used as initializer to @gptr.8,
;          and @gptr.8 is used within non-kernel function @f1. Hence @lds.7 is lowered.
;.

; CHECK: %llvm.amdgcn.module.lds.t = type { [3 x float], [1 x float], [2 x float] }

; CHECK: @lds.1 = addrspace(3) global i16 undef, align 2
; CHECK: @lds.2 = addrspace(3) global i32 undef, align 4
; CHECK: @lds.3 = addrspace(3) global i64 undef, align 8
; CHECK: @lds.4 = addrspace(3) global float undef, align 4
; CHECK-NOT: @lds.5
; CHECK-NOT: @lds.6
; CHECK-NOT: @lds.7
@lds.1 = addrspace(3) global i16 undef, align 2
@lds.2 = addrspace(3) global i32 undef, align 4
@lds.3 = addrspace(3) global i64 undef, align 8
@lds.4 = addrspace(3) global float undef, align 4
@lds.5 = addrspace(3) global [1 x float] undef, align 4
@lds.6 = addrspace(3) global [2 x float] undef, align 8
@lds.7 = addrspace(3) global [3 x float] undef, align 16

; CHECK: @gptr.3 = addrspace(1) global i64* addrspacecast (i64 addrspace(3)* @lds.3 to i64*), align 8
; CHECK: @gptr.4 = addrspace(1) global i64* addrspacecast (i64 addrspace(3)* bitcast (float addrspace(3)* @lds.4 to i64 addrspace(3)*) to i64*), align 8
; CHECK: @gptr.5 = addrspace(1) global i64* addrspacecast (i64 addrspace(3)* bitcast ([1 x float] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 1) to i64 addrspace(3)*) to i64*), align 8
; CHECK: @gptr.6 = addrspace(1) global i64* addrspacecast (i64 addrspace(3)* bitcast ([2 x float] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 2) to i64 addrspace(3)*) to i64*), align 8
; CHECK: @gptr.7 = addrspace(1) global i64* addrspacecast (i64 addrspace(3)* bitcast (%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds to i64 addrspace(3)*) to i64*), align 8
; CHECK: @gptr.8 = addrspace(1) global i64** addrspacecast (i64* addrspace(1)* @gptr.7 to i64**), align 8
@gptr.3 = addrspace(1) global i64* addrspacecast (i64 addrspace(3)* @lds.3 to i64*), align 8
@gptr.4 = addrspace(1) global i64* addrspacecast (float addrspace(3)* @lds.4 to i64*), align 8
@gptr.5 = addrspace(1) global i64* addrspacecast ([1 x float] addrspace(3)* @lds.5 to i64*), align 8
@gptr.6 = addrspace(1) global i64* addrspacecast ([2 x float] addrspace(3)* @lds.6 to i64*), align 8
@gptr.7 = addrspace(1) global i64* addrspacecast ([3 x float] addrspace(3)* @lds.7 to i64*), align 8
@gptr.8 = addrspace(1) global i64** addrspacecast (i64* addrspace(1)* @gptr.7 to i64**), align 8

; CHECK: @llvm.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (i16 addrspace(3)* @lds.1 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"
; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t undef, align 16
; CHECK: @llvm.compiler.used = appending global [5 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (i32 addrspace(3)* @lds.2 to i8 addrspace(3)*) to i8*), i8* addrspacecast (i8 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.4 to i8 addrspace(1)*) to i8*), i8* addrspacecast (i8 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.5 to i8 addrspace(1)*) to i8*), i8* addrspacecast (i8 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.6 to i8 addrspace(1)*) to i8*), i8* addrspacecast (i8 addrspace(3)* bitcast (%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds to i8 addrspace(3)*) to i8*)], section "llvm.metadata"
@llvm.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (i16 addrspace(3)* @lds.1 to i8 addrspace(3)*) to i8*)], section "llvm.metadata"
@llvm.compiler.used = appending global [4 x i8*] [i8* addrspacecast (i8 addrspace(3)* bitcast (i32 addrspace(3)* @lds.2 to i8 addrspace(3)*) to i8*), i8* addrspacecast (i8 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.4 to i8 addrspace(1)*) to i8*), i8* addrspacecast (i8 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.5 to i8 addrspace(1)*) to i8*), i8* addrspacecast (i8 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.6 to i8 addrspace(1)*) to i8*)], section "llvm.metadata"

; CHECK-LABEL: @f1()
; CHECK:   %ld = load i64**, i64** addrspace(1)* @gptr.8, align 8
; CHECK:   ret void
define void @f1() {
  %ld = load i64**, i64** addrspace(1)* @gptr.8
  ret void
}

; CHECK-LABEL: @f0()
; CHECK:   %ld = load i32, i32* inttoptr (i64 add (i64 ptrtoint (i32* addrspacecast (i32 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.6 to i32 addrspace(1)*) to i32*) to i64), i64 ptrtoint (i32* addrspacecast (i32
; CHECK: addrspace(1)* bitcast (i64* addrspace(1)* @gptr.6 to i32 addrspace(1)*) to i32*) to i64)) to i32*), align 4
; CHECK:   ret void
define void @f0() {
  %ld = load i32, i32* inttoptr (i64 add (i64 ptrtoint (i32* addrspacecast (i32 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.6 to i32 addrspace(1)*) to i32*) to i64), i64 ptrtoint (i32* addrspacecast (i32 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.6 to i32 addrspace(1)*) to i32*) to i64)) to i32*), align 4
  ret void
}

; CHECK-LABEL: @k0()
; CHECK:   call void @llvm.donothing() [ "ExplicitUse"(%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds) ]
; CHECK:   %ld = load i32, i32* inttoptr (i64 add (i64 ptrtoint (i32* addrspacecast (i32 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.5 to i32 addrspace(1)*) to i32*) to i64), i64 ptrtoint (i32* addrspacecast (i32
; CHECK: addrspace(1)* bitcast (i64* addrspace(1)* @gptr.5 to i32 addrspace(1)*) to i32*) to i64)) to i32*), align 4
; CHECK:   ret void
define amdgpu_kernel void @k0() {
  %ld = load i32, i32* inttoptr (i64 add (i64 ptrtoint (i32* addrspacecast (i32 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.5 to i32 addrspace(1)*) to i32*) to i64), i64 ptrtoint (i32* addrspacecast (i32 addrspace(1)* bitcast (i64* addrspace(1)* @gptr.5 to i32 addrspace(1)*) to i32*) to i64)) to i32*), align 4
  ret void
}

; CHECK-LABEL: @k1()
; CHECK:   call void @llvm.donothing() [ "ExplicitUse"(%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds) ]
; CHECK:   ret void
define amdgpu_kernel void @k1() {
  ret void
}
