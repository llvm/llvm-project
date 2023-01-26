; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s

@lds.size.1.align.1 = internal unnamed_addr addrspace(3) global [1 x i8] undef, align 1
@lds.size.2.align.2 = internal unnamed_addr addrspace(3) global [2 x i8] undef, align 2
@lds.size.4.align.4 = internal unnamed_addr addrspace(3) global [4 x i8] undef, align 4
@lds.size.16.align.16 = internal unnamed_addr addrspace(3) global [16 x i8] undef, align 16

; CHECK: %llvm.amdgcn.kernel.k0.lds.t = type { [16 x i8], [4 x i8], [2 x i8], [1 x i8] }
; CHECK: %llvm.amdgcn.kernel.k1.lds.t = type { [16 x i8], [4 x i8], [2 x i8] }

;.
; CHECK: @lds.k2 = addrspace(3) global [1 x i8] undef, align 1
; CHECK: @llvm.amdgcn.kernel.k0.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k0.lds.t undef, align 16
; CHECK: @llvm.amdgcn.kernel.k1.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k1.lds.t undef, align 16
;.
define amdgpu_kernel void @k0() {
; CHECK-LABEL: @k0(
; CHECK-NEXT: store i8 1, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, i32 0, i32 3), align 2, !alias.scope !0, !noalias !3
; CHECK-NEXT: store i8 2, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, i32 0, i32 2), align 4, !alias.scope !7, !noalias !8
; CHECK-NEXT: store i8 4, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, i32 0, i32 1), align 16, !alias.scope !9, !noalias !10
; CHECK-NEXT: store i8 16, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, align 16, !alias.scope !11, !noalias !12
; CHECK-NEXT: ret void
  store i8 1, ptr addrspace(3) @lds.size.1.align.1, align 1

  store i8 2, ptr addrspace(3) @lds.size.2.align.2, align 2

  store i8 4, ptr addrspace(3) @lds.size.4.align.4, align 4

  store i8 16, ptr addrspace(3) @lds.size.16.align.16, align 16

  ret void
}

define amdgpu_kernel void @k1() {
; CHECK-LABEL: @k1(
; CHECK-NEXT: store i8 2, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds, i32 0, i32 2), align 4, !alias.scope !13, !noalias !16
; CHECK-NEXT: store i8 4, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds, i32 0, i32 1), align 16, !alias.scope !19, !noalias !20
; CHECK-NEXT: store i8 16, ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds, align 16, !alias.scope !21, !noalias !22
; CHECK-NEXT: ret void
;
  store i8 2, ptr addrspace(3) @lds.size.2.align.2, align 2

  store i8 4, ptr addrspace(3) @lds.size.4.align.4, align 4

  store i8 16, ptr addrspace(3) @lds.size.16.align.16, align 16

  ret void
}

; Do not lower LDS for graphics shaders.

@lds.k2 = addrspace(3) global [1 x i8] undef, align 1

define amdgpu_ps void @k2() {
; CHECK-LABEL: @k2(
; CHECK-NEXT:    store i8 1, ptr addrspace(3) @lds.k2, align 1
; CHECK-NEXT:    ret void
;
  store i8 1, ptr addrspace(3) @lds.k2, align 1

  ret void
}
