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
; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t undef, align 8, !absolute_symbol !0
; CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr addrspacecast (ptr addrspace(3) @llvm.amdgcn.module.lds to ptr)], section "llvm.metadata"
; CHECK: @llvm.amdgcn.kernel.k0.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k0.lds.t undef, align 16, !absolute_symbol !0
; CHECK: @llvm.amdgcn.kernel.k1.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k1.lds.t undef, align 16, !absolute_symbol !0
; CHECK: @llvm.amdgcn.kernel.k2.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k2.lds.t undef, align 2, !absolute_symbol !0
; CHECK: @llvm.amdgcn.kernel.k3.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k3.lds.t undef, align 4, !absolute_symbol !0
;.
define amdgpu_kernel void @k0() #0 {
; CHECK-LABEL: @k0() #0
; CHECK-NEXT: store i8 1, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, i32 0, i32 3), align 2, !alias.scope !1, !noalias !4
; CHECK-NEXT: store i8 2, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, i32 0, i32 2), align 4, !alias.scope !8, !noalias !9
; CHECK-NEXT: store i8 4, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k0.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, i32 0, i32 1), align 16, !alias.scope !10, !noalias !11
; CHECK-NEXT: store i8 16, ptr addrspace(3) @llvm.amdgcn.kernel.k0.lds, align 16, !alias.scope !12, !noalias !13
; CHECK-NEXT:    ret void
  store i8 1, ptr addrspace(3) @lds.size.1.align.1, align 1

  store i8 2, ptr addrspace(3) @lds.size.2.align.2, align 2

  store i8 4, ptr addrspace(3) @lds.size.4.align.4, align 4

  store i8 16, ptr addrspace(3) @lds.size.16.align.16, align 16

  ret void
}

define amdgpu_kernel void @k1() #0 {
; CHECK-LABEL: @k1() #1
; CHECK-NEXT: store i8 2, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds, i32 0, i32 2), align 4, !alias.scope !14, !noalias !17
; CHECK-NEXT: store i8 4, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.kernel.k1.lds.t, ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds, i32 0, i32 1), align 16, !alias.scope !20, !noalias !21
; CHECK-NEXT: store i8 16, ptr addrspace(3) @llvm.amdgcn.kernel.k1.lds, align 16, !alias.scope !22, !noalias !23
; CHECK-NEXT:    ret void
;
  store i8 2, ptr addrspace(3) @lds.size.2.align.2, align 2

  store i8 4, ptr addrspace(3) @lds.size.4.align.4, align 4

  store i8 16, ptr addrspace(3) @lds.size.16.align.16, align 16

  ret void
}

define amdgpu_kernel void @k2() #0 {
; CHECK-LABEL: @k2() #2
; CHECK-NEXT:    store i8 2, ptr addrspace(3) @llvm.amdgcn.kernel.k2.lds, align 2
; CHECK-NEXT:    ret void
;
  store i8 2, ptr addrspace(3) @lds.size.2.align.2, align 2

  ret void
}

define amdgpu_kernel void @k3() #0 {
; CHECK-LABEL: @k3() #3
; CHECK-NEXT:    store i8 4, ptr addrspace(3) @llvm.amdgcn.kernel.k3.lds, align 4
; CHECK-NEXT:    ret void
;
  store i8 4, ptr addrspace(3) @lds.size.4.align.4, align 4

  ret void
}

; CHECK-LABEL: @calls_f0() #4
define amdgpu_kernel void @calls_f0() {
  call void @f0()
  ret void
}

define void @f0() {
; CHECK-LABEL: define void @f0()
; CHECK-NEXT: store i8 1, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.module.lds.t, ptr addrspace(3) @llvm.amdgcn.module.lds, i32 0, i32 1), align 8, !noalias !24
; CHECK-NEXT: store i8 8, ptr addrspace(3) @llvm.amdgcn.module.lds, align 8, !noalias !24
; CHECK-NEXT: ret void
  store i8 1, ptr addrspace(3) @lds.size.1.align.1, align 1

  store i8 8, ptr addrspace(3) @lds.size.8.align.8, align 4

  ret void
}

; CHECK: attributes #0 = { "amdgpu-lds-size"="23" }
; CHECK: attributes #1 = { "amdgpu-lds-size"="22" }
; CHECK: attributes #2 = { "amdgpu-lds-size"="2" }
; CHECK: attributes #3 = { "amdgpu-lds-size"="4" }
; CHECK: attributes #4 = { "amdgpu-lds-size"="9" }

; CHECK: !0 = !{i64 0, i64 1}
