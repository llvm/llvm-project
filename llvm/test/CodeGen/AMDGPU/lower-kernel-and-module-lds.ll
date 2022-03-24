; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

@lds.size.1.align.1 = internal unnamed_addr addrspace(3) global [1 x i8] undef, align 1
@lds.size.2.align.2 = internal unnamed_addr addrspace(3) global [2 x i8] undef, align 2
@lds.size.4.align.4 = internal unnamed_addr addrspace(3) global [4 x i8] undef, align 4
@lds.size.8.align.8 = internal unnamed_addr addrspace(3) global [8 x i8] undef, align 8
@lds.size.16.align.16 = internal unnamed_addr addrspace(3) global [16 x i8] undef, align 16

; CHECK: %llvm.amdgcn.module.lds.t = type { [8 x i8], [1 x i8] }
; CHECK: %llvm.amdgcn.kernel.k0.lds.t = type { [16 x i8], [4 x i8], [2 x i8] }
; CHECK: %llvm.amdgcn.kernel.k1.lds.t = type { [16 x i8], [4 x i8], [2 x i8] }
; CHECK: %llvm.amdgcn.kernel..lds.t = type { [2 x i8] }
; CHECK: %llvm.amdgcn.kernel..lds.t.0 = type { [4 x i8] }

;.
; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t undef, align 8
; CHECK: @llvm.compiler.used = appending global [1 x i8*] [i8* addrspacecast (i8 addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 0, i32 0) to i8*)], section "llvm.metadata"
; CHECK: @llvm.amdgcn.kernel.k0.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k0.lds.t undef, align 16
; CHECK: @llvm.amdgcn.kernel.k1.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k1.lds.t undef, align 16
; CHECK: @llvm.amdgcn.kernel..lds = internal addrspace(3) global %llvm.amdgcn.kernel..lds.t undef, align 2
; CHECK: @llvm.amdgcn.kernel..lds.1 = internal addrspace(3) global %llvm.amdgcn.kernel..lds.t.0 undef, align 4
;.
define amdgpu_kernel void @k0() {
; CHECK-LABEL: @k0(
; CHECK-NEXT:    call void @llvm.donothing() [ "ExplicitUse"(%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds) ]
; CHECK-NEXT:    %lds.size.1.align.1.bc = bitcast [1 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 1) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 1, i8 addrspace(3)* %lds.size.1.align.1.bc, align 8
; CHECK-NEXT:    %lds.size.2.align.2.bc = bitcast [2 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, %llvm.amdgcn.kernel.k0.lds.t addrspace(3)* @llvm.amdgcn.kernel.k0.lds, i32 0, i32 2) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 2, i8 addrspace(3)* %lds.size.2.align.2.bc, align 4
; CHECK-NEXT:    %lds.size.4.align.4.bc = bitcast [4 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, %llvm.amdgcn.kernel.k0.lds.t addrspace(3)* @llvm.amdgcn.kernel.k0.lds, i32 0, i32 1) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 4, i8 addrspace(3)* %lds.size.4.align.4.bc, align 16
; CHECK-NEXT:    %lds.size.16.align.16.bc = bitcast [16 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, %llvm.amdgcn.kernel.k0.lds.t addrspace(3)* @llvm.amdgcn.kernel.k0.lds, i32 0, i32 0) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 16, i8 addrspace(3)* %lds.size.16.align.16.bc, align 16
; CHECK-NEXT:    ret void
;
  %lds.size.1.align.1.bc = bitcast [1 x i8] addrspace(3)* @lds.size.1.align.1 to i8 addrspace(3)*
  store i8 1, i8 addrspace(3)* %lds.size.1.align.1.bc, align 1

  %lds.size.2.align.2.bc = bitcast [2 x i8] addrspace(3)* @lds.size.2.align.2 to i8 addrspace(3)*
  store i8 2, i8 addrspace(3)* %lds.size.2.align.2.bc, align 2

  %lds.size.4.align.4.bc = bitcast [4 x i8] addrspace(3)* @lds.size.4.align.4 to i8 addrspace(3)*
  store i8 4, i8 addrspace(3)* %lds.size.4.align.4.bc, align 4

  %lds.size.16.align.16.bc = bitcast [16 x i8] addrspace(3)* @lds.size.16.align.16 to i8 addrspace(3)*
  store i8 16, i8 addrspace(3)* %lds.size.16.align.16.bc, align 16

  ret void
}

define amdgpu_kernel void @k1() {
; CHECK-LABEL: @k1(
; CHECK-NEXT:    call void @llvm.donothing() [ "ExplicitUse"(%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds) ]
; CHECK-NEXT:    %lds.size.2.align.2.bc = bitcast [2 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, %llvm.amdgcn.kernel.k1.lds.t addrspace(3)* @llvm.amdgcn.kernel.k1.lds, i32 0, i32 2) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 2, i8 addrspace(3)* %lds.size.2.align.2.bc, align 4
; CHECK-NEXT:    %lds.size.4.align.4.bc = bitcast [4 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, %llvm.amdgcn.kernel.k1.lds.t addrspace(3)* @llvm.amdgcn.kernel.k1.lds, i32 0, i32 1) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 4, i8 addrspace(3)* %lds.size.4.align.4.bc, align 16
; CHECK-NEXT:    %lds.size.16.align.16.bc = bitcast [16 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, %llvm.amdgcn.kernel.k1.lds.t addrspace(3)* @llvm.amdgcn.kernel.k1.lds, i32 0, i32 0) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 16, i8 addrspace(3)* %lds.size.16.align.16.bc, align 16
; CHECK-NEXT:    ret void
;
  %lds.size.2.align.2.bc = bitcast [2 x i8] addrspace(3)* @lds.size.2.align.2 to i8 addrspace(3)*
  store i8 2, i8 addrspace(3)* %lds.size.2.align.2.bc, align 2

  %lds.size.4.align.4.bc = bitcast [4 x i8] addrspace(3)* @lds.size.4.align.4 to i8 addrspace(3)*
  store i8 4, i8 addrspace(3)* %lds.size.4.align.4.bc, align 4

  %lds.size.16.align.16.bc = bitcast [16 x i8] addrspace(3)* @lds.size.16.align.16 to i8 addrspace(3)*
  store i8 16, i8 addrspace(3)* %lds.size.16.align.16.bc, align 16

  ret void
}

define amdgpu_kernel void @0() {
; CHECK-LABEL: @0(
; CHECK-NEXT:    call void @llvm.donothing() [ "ExplicitUse"(%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds) ]
; CHECK-NEXT:    %lds.size.2.align.2.bc = bitcast [2 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel..lds.t, %llvm.amdgcn.kernel..lds.t addrspace(3)* @llvm.amdgcn.kernel..lds, i32 0, i32 0) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 2, i8 addrspace(3)* %lds.size.2.align.2.bc, align 2
; CHECK-NEXT:    ret void
;
  %lds.size.2.align.2.bc = bitcast [2 x i8] addrspace(3)* @lds.size.2.align.2 to i8 addrspace(3)*
  store i8 2, i8 addrspace(3)* %lds.size.2.align.2.bc, align 2

  ret void
}

define amdgpu_kernel void @1() {
; CHECK-LABEL: @1(
; CHECK-NEXT:    call void @llvm.donothing() [ "ExplicitUse"(%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds) ]
; CHECK-NEXT:    %lds.size.4.align.4.bc = bitcast [4 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.kernel..lds.t.0, %llvm.amdgcn.kernel..lds.t.0 addrspace(3)* @llvm.amdgcn.kernel..lds.1, i32 0, i32 0) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 4, i8 addrspace(3)* %lds.size.4.align.4.bc, align 4
; CHECK-NEXT:    ret void
;
  %lds.size.4.align.4.bc = bitcast [4 x i8] addrspace(3)* @lds.size.4.align.4 to i8 addrspace(3)*
  store i8 4, i8 addrspace(3)* %lds.size.4.align.4.bc, align 4

  ret void
}

define void @f0() {
; CHECK-LABEL: @f0(
; CHECK-NEXT:    %lds.size.1.align.1.bc = bitcast [1 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 1) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 1, i8 addrspace(3)* %lds.size.1.align.1.bc, align 8
; CHECK-NEXT:    %lds.size.8.align.8.bc = bitcast [8 x i8] addrspace(3)* getelementptr inbounds (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds, i32 0, i32 0) to i8 addrspace(3)*
; CHECK-NEXT:    store i8 8, i8 addrspace(3)* %lds.size.8.align.8.bc, align 8
; CHECK-NEXT:    ret void
;
  %lds.size.1.align.1.bc = bitcast [1 x i8] addrspace(3)* @lds.size.1.align.1 to i8 addrspace(3)*
  store i8 1, i8 addrspace(3)* %lds.size.1.align.1.bc, align 1

  %lds.size.8.align.8.bc = bitcast [8 x i8] addrspace(3)* @lds.size.8.align.8 to i8 addrspace(3)*
  store i8 8, i8 addrspace(3)* %lds.size.8.align.8.bc, align 4

  ret void
}
;.
; CHECK: attributes #0 = { nocallback nofree nosync nounwind readnone willreturn }
;.
