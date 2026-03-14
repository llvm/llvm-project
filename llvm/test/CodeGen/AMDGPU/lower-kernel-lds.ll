; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s

@lds.size.1.align.1 = internal unnamed_addr addrspace(3) global [1 x i8] poison, align 1
@lds.size.2.align.2 = internal unnamed_addr addrspace(3) global [2 x i8] poison, align 2
@lds.size.4.align.4 = internal unnamed_addr addrspace(3) global [4 x i8] poison, align 4
@lds.size.16.align.16 = internal unnamed_addr addrspace(3) global [16 x i8] poison, align 16

; CHECK: %llvm.amdgcn.kernel.k0.lds.t = type { [16 x i8], [4 x i8], [2 x i8], [1 x i8] }
; CHECK: %llvm.amdgcn.kernel.k1.lds.t = type { [16 x i8], [4 x i8], [2 x i8] }

;.
; CHECK: @lds.k2 = addrspace(3) global [1 x i8] poison, align 1
; CHECK: @llvm.amdgcn.kernel.k0.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k0.lds.t poison, align 16, !absolute_symbol !0
; CHECK: @llvm.amdgcn.kernel.k1.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k1.lds.t poison, align 16, !absolute_symbol !0
;.
define amdgpu_kernel void @k0() {
; CHECK-LABEL: @k0(
; CHECK-NEXT: store i8 1, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, i32 0, i32 3), align 2, !alias.scope !1, !noalias !4
; CHECK-NEXT: store i8 2, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, i32 0, i32 2), align 4, !alias.scope !8, !noalias !9
; CHECK-NEXT: store i8 4, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, i32 0, i32 1), align 16, !alias.scope !10, !noalias !11
; CHECK-NEXT: store i8 16, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, align 16, !alias.scope !12, !noalias !13
; CHECK-NEXT: ret void
  store i8 1, ptr addrspace(3) @lds.size.1.align.1, align 1

  store i8 2, ptr addrspace(3) @lds.size.2.align.2, align 2

  store i8 4, ptr addrspace(3) @lds.size.4.align.4, align 4

  store i8 16, ptr addrspace(3) @lds.size.16.align.16, align 16

  ret void
}

define amdgpu_kernel void @k1() {
; CHECK-LABEL: @k1(
; CHECK-NEXT: store i8 2, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds, i32 0, i32 2), align 4, !alias.scope !14, !noalias !17
; CHECK-NEXT: store i8 4, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds, i32 0, i32 1), align 16, !alias.scope !20, !noalias !21
; CHECK-NEXT: store i8 16, ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds, align 16, !alias.scope !22, !noalias !23
; CHECK-NEXT: ret void
;
  store i8 2, ptr addrspace(3) @lds.size.2.align.2, align 2

  store i8 4, ptr addrspace(3) @lds.size.4.align.4, align 4

  store i8 16, ptr addrspace(3) @lds.size.16.align.16, align 16

  ret void
}

; Do not lower LDS for graphics shaders.

@lds.k2 = addrspace(3) global [1 x i8] poison, align 1

define amdgpu_ps void @k2() {
; CHECK-LABEL: @k2(
; CHECK-NEXT:    store i8 1, ptr addrspace(3) @lds.k2, align 1
; CHECK-NEXT:    ret void
;
  store i8 1, ptr addrspace(3) @lds.k2, align 1

  ret void
}

; CHECK: !0 = !{i32 0, i32 1}
