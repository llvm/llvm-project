; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s

@lds.size.1.align.1 = internal unnamed_addr addrspace(3) global [1 x i8] undef, align 1
@lds.size.2.align.2 = internal unnamed_addr addrspace(3) global [2 x i8] undef, align 2
@lds.size.4.align.4 = internal unnamed_addr addrspace(3) global [4 x i8] undef, align 4
@lds.size.8.align.8 = internal unnamed_addr addrspace(3) global [8 x i8] undef, align 8
@lds.size.16.align.16 = internal unnamed_addr addrspace(3) global [16 x i8] undef, align 16

; CHECK: %llvm.amdgcn.module.lds.t = type { [8 x i8], [1 x i8] }
; CHECK: %llvm.amdgcn.kernel.k0.lds.t = type { [16 x i8], [4 x i8], [2 x i8], [1 x i8] }
; CHECK: %llvm.amdgcn.kernel.k1.lds.t = type { [16 x i8], [4 x i8], [2 x i8] }
; CHECK: %llvm.amdgcn.kernel.k2.lds.t = type { [2 x i8] }
; CHECK: %llvm.amdgcn.kernel.k3.lds.t = type { [4 x i8] }

;.
; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t undef, align 8
; CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @llvm.amdgcn.module.lds to ptr)], section "llvm.metadata"
; CHECK: @llvm.amdgcn.kernel.k0.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k0.lds.t undef, align 16
; CHECK: @llvm.amdgcn.kernel.k1.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k1.lds.t undef, align 16
; CHECK: @llvm.amdgcn.kernel.k2.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k2.lds.t undef, align 2
; CHECK: @llvm.amdgcn.kernel.k3.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k3.lds.t undef, align 4
;.
define amdgpu_kernel void @k0() #0 {
; CHECK-LABEL: @k0(
; CHECK-NEXT: store i8 1, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, i32 0, i32 3), align 2, !alias.scope !0, !noalias !3
; CHECK-NEXT: store i8 2, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, i32 0, i32 2), align 4, !alias.scope !7, !noalias !8
; CHECK-NEXT: store i8 4, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, i32 0, i32 1), align 16, !alias.scope !9, !noalias !10
; CHECK-NEXT: store i8 16, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, align 16, !alias.scope !11, !noalias !12
; CHECK-NEXT:    ret void
  store i8 1, ptr addrspace(3) @lds.size.1.align.1, align 1

  store i8 2, ptr addrspace(3) @lds.size.2.align.2, align 2

  store i8 4, ptr addrspace(3) @lds.size.4.align.4, align 4

  store i8 16, ptr addrspace(3) @lds.size.16.align.16, align 16

  ret void
}

define amdgpu_kernel void @k1() #0 {
; CHECK-LABEL: @k1(
; CHECK-NEXT: store i8 2, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds, i32 0, i32 2), align 4, !alias.scope !13, !noalias !16
; CHECK-NEXT: store i8 4, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds, i32 0, i32 1), align 16, !alias.scope !19, !noalias !20
; CHECK-NEXT: store i8 16, ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds, align 16, !alias.scope !21, !noalias !22
; CHECK-NEXT:    ret void
;
  store i8 2, ptr addrspace(3) @lds.size.2.align.2, align 2

  store i8 4, ptr addrspace(3) @lds.size.4.align.4, align 4

  store i8 16, ptr addrspace(3) @lds.size.16.align.16, align 16

  ret void
}

define amdgpu_kernel void @k2() #0 {
; CHECK-LABEL: @k2(
; CHECK-NEXT:    store i8 2, ptr addrspace(3) @llvm.amdgcn.kernel.k2.lds, align 2
; CHECK-NEXT:    ret void
;
  store i8 2, ptr addrspace(3) @lds.size.2.align.2, align 2

  ret void
}

define amdgpu_kernel void @k3() #0 {
; CHECK-LABEL: @k3(
; CHECK-NEXT:    store i8 4, ptr addrspace(3) @llvm.amdgcn.kernel.k3.lds, align 4
; CHECK-NEXT:    ret void
;
  store i8 4, ptr addrspace(3) @lds.size.4.align.4, align 4

  ret void
}


define amdgpu_kernel void @calls_f0() {
  call void @f0()
  ret void
}

define void @f0() {
; CHECK-LABEL: define void @f0(
; CHECK-NEXT: store i8 1, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.module.lds.t, ptr addrspace(3) @llvm.amdgcn.module.lds, i32 0, i32 1), align 8, !noalias !23
; CHECK-NEXT: store i8 8, ptr addrspace(3) @llvm.amdgcn.module.lds, align 8, !noalias !23
; CHECK-NEXT: ret void
  store i8 1, ptr addrspace(3) @lds.size.1.align.1, align 1

  store i8 8, ptr addrspace(3) @lds.size.8.align.8, align 4

  ret void
}

attributes #0 = { "amdgpu-elide-module-lds" }
; CHECK: attributes #0 = { "amdgpu-elide-module-lds" }
