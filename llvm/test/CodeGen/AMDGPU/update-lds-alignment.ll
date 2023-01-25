; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

; Properly aligned, same size as alignment.
; CHECK: %llvm.amdgcn.kernel.k0.lds.t = type { [16 x i8], [8 x i8], [4 x i8], [2 x i8], [1 x i8] }

; Different properly aligned values, but same size of 1.
; CHECK: %llvm.amdgcn.kernel.k1.lds.t = type { [1 x i8], [1 x i8], [1 x i8], [1 x i8], [1 x i8], [3 x i8], [1 x i8] }

; All are under-aligned, requires to fix each on different alignment boundary.
; CHECK: %llvm.amdgcn.kernel.k2.lds.t = type { [9 x i8], [1 x i8], [2 x i8], [3 x i8], [1 x i8], [5 x i8] }

; All LDS are underaligned, requires to allocate on 8 byte boundary
; CHECK: %llvm.amdgcn.kernel.k3.lds.t = type { [7 x i8], [1 x i8], [7 x i8], [1 x i8], [6 x i8], [2 x i8], [5 x i8] }

; All LDS are underaligned, requires to allocate on 16 byte boundary
; CHECK: %llvm.amdgcn.kernel.k4.lds.t = type { [12 x i8], [4 x i8], [11 x i8], [5 x i8], [10 x i8], [6 x i8], [9 x i8] }

; All LDS are properly aligned on 16 byte boundary, but they are of different size.
; CHECK: %llvm.amdgcn.kernel.k5.lds.t = type { [20 x i8], [12 x i8], [19 x i8], [13 x i8], [18 x i8], [14 x i8], [17 x i8] }

; CHECK: @llvm.amdgcn.kernel.k0.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k0.lds.t undef, align 16
; CHECK: @llvm.amdgcn.kernel.k1.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k1.lds.t undef, align 16
; CHECK: @llvm.amdgcn.kernel.k2.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k2.lds.t undef, align 16
; CHECK: @llvm.amdgcn.kernel.k3.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k3.lds.t undef, align 8
; CHECK: @llvm.amdgcn.kernel.k4.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k4.lds.t undef, align 16
; CHECK: @llvm.amdgcn.kernel.k5.lds = internal addrspace(3) global %llvm.amdgcn.kernel.k5.lds.t undef, align 16


; Properly aligned, same size as alignment.
; CHECK-NOT: @k0.lds.size.1.align.1
; CHECK-NOT: @k0.lds.size.2.align.2
; CHECK-NOT: @k0.lds.size.4.align.4
; CHECK-NOT: @k0.lds.size.8.align.8
; CHECK-NOT: @k0.lds.size.16.align.16
@k0.lds.size.1.align.1 = internal unnamed_addr addrspace(3) global [1 x i8] undef, align 1
@k0.lds.size.2.align.2 = internal unnamed_addr addrspace(3) global [2 x i8] undef, align 2
@k0.lds.size.4.align.4 = internal unnamed_addr addrspace(3) global [4 x i8] undef, align 4
@k0.lds.size.8.align.8 = internal unnamed_addr addrspace(3) global [8 x i8] undef, align 8
@k0.lds.size.16.align.16 = internal unnamed_addr addrspace(3) global [16 x i8] undef, align 16

define amdgpu_kernel void @k0() {
   store i8 1, ptr addrspace(3) @k0.lds.size.1.align.1, align 1

   store i8 2, ptr addrspace(3) @k0.lds.size.2.align.2, align 2

   store i8 3, ptr addrspace(3) @k0.lds.size.4.align.4, align 4

   store i8 4, ptr addrspace(3) @k0.lds.size.8.align.8, align 8

   store i8 5, ptr addrspace(3) @k0.lds.size.16.align.16, align 16

  ret void
}

; Different properly aligned values, but same size of 1.
; CHECK-NOT: @k1.lds.size.1.align.1
; CHECK-NOT: @k1.lds.size.1.align.2
; CHECK-NOT: @k1.lds.size.1.align.4
; CHECK-NOT: @k1.lds.size.1.align.8
; CHECK-NOT: @k1.lds.size.1.align.16
@k1.lds.size.1.align.1 = internal unnamed_addr addrspace(3) global [1 x i8] undef, align 1
@k1.lds.size.1.align.2 = internal unnamed_addr addrspace(3) global [1 x i8] undef, align 2
@k1.lds.size.1.align.4 = internal unnamed_addr addrspace(3) global [1 x i8] undef, align 4
@k1.lds.size.1.align.8 = internal unnamed_addr addrspace(3) global [1 x i8] undef, align 8
@k1.lds.size.1.align.16 = internal unnamed_addr addrspace(3) global [1 x i8] undef, align 16

define amdgpu_kernel void @k1() {
   store i8 1, ptr addrspace(3) @k1.lds.size.1.align.1, align 1

   store i8 2, ptr addrspace(3) @k1.lds.size.1.align.2, align 2

   store i8 3, ptr addrspace(3) @k1.lds.size.1.align.4, align 4

   store i8 4, ptr addrspace(3) @k1.lds.size.1.align.8, align 8

   store i8 5, ptr addrspace(3) @k1.lds.size.1.align.16, align 16

  ret void
}

; All are under-aligned, requires to fix each on different alignment boundary.
; CHECK-NOT: @k2.lds.size.2.align.1
; CHECK-NOT: @k2.lds.size.3.align.2
; CHECK-NOT: @k2.lds.size.5.align.4
; CHECK-NOT: @k2.lds.size.9.align.8
@k2.lds.size.2.align.1 = internal unnamed_addr addrspace(3) global [2 x i8] undef, align 1
@k2.lds.size.3.align.2 = internal unnamed_addr addrspace(3) global [3 x i8] undef, align 2
@k2.lds.size.5.align.4 = internal unnamed_addr addrspace(3) global [5 x i8] undef, align 4
@k2.lds.size.9.align.8 = internal unnamed_addr addrspace(3) global [9 x i8] undef, align 8

define amdgpu_kernel void @k2() {
   store i8 1, ptr addrspace(3) @k2.lds.size.2.align.1, align 1

   store i8 2, ptr addrspace(3) @k2.lds.size.3.align.2, align 2

   store i8 3, ptr addrspace(3) @k2.lds.size.5.align.4, align 4

   store i8 4, ptr addrspace(3) @k2.lds.size.9.align.8, align 8

  ret void
}

; All LDS are underaligned, requires to allocate on 8 byte boundary
; CHECK-NOT: @k3.lds.size.5.align.2
; CHECK-NOT: @k3.lds.size.6.align.2
; CHECK-NOT: @k3.lds.size.7.align.2
; CHECK-NOT: @k3.lds.size.7.align.4
@k3.lds.size.5.align.2 = internal unnamed_addr addrspace(3) global [5 x i8] undef, align 2
@k3.lds.size.6.align.2 = internal unnamed_addr addrspace(3) global [6 x i8] undef, align 2
@k3.lds.size.7.align.2 = internal unnamed_addr addrspace(3) global [7 x i8] undef, align 2
@k3.lds.size.7.align.4 = internal unnamed_addr addrspace(3) global [7 x i8] undef, align 4

define amdgpu_kernel void @k3() {
   store i8 1, ptr addrspace(3) @k3.lds.size.5.align.2, align 2

   store i8 2, ptr addrspace(3) @k3.lds.size.6.align.2, align 2

   store i8 3, ptr addrspace(3) @k3.lds.size.7.align.2, align 2

   store i8 4, ptr addrspace(3) @k3.lds.size.7.align.4, align 4

  ret void
}

; All LDS are underaligned, requires to allocate on 16 byte boundary
; CHECK-NOT: @k4.lds.size.9.align.1
; CHECK-NOT: @k4.lds.size.10.align.2
; CHECK-NOT: @k4.lds.size.11.align.4
; CHECK-NOT: @k4.lds.size.12.align.8
@k4.lds.size.9.align.1 = internal unnamed_addr addrspace(3) global [9 x i8] undef, align 1
@k4.lds.size.10.align.2 = internal unnamed_addr addrspace(3) global [10 x i8] undef, align 2
@k4.lds.size.11.align.4 = internal unnamed_addr addrspace(3) global [11 x i8] undef, align 4
@k4.lds.size.12.align.8 = internal unnamed_addr addrspace(3) global [12 x i8] undef, align 8

define amdgpu_kernel void @k4() {
   store i8 1, ptr addrspace(3) @k4.lds.size.9.align.1, align 1

   store i8 2, ptr addrspace(3) @k4.lds.size.10.align.2, align 2

   store i8 3, ptr addrspace(3) @k4.lds.size.11.align.4, align 4

   store i8 4, ptr addrspace(3) @k4.lds.size.12.align.8, align 8

  ret void
}

; CHECK-NOT: @k5.lds.size.17.align.16
; CHECK-NOT: @k5.lds.size.18.align.16
; CHECK-NOT: @k5.lds.size.19.align.16
; CHECK-NOT: @k5.lds.size.20.align.16
@k5.lds.size.17.align.16 = internal unnamed_addr addrspace(3) global [17 x i8] undef, align 16
@k5.lds.size.18.align.16 = internal unnamed_addr addrspace(3) global [18 x i8] undef, align 16
@k5.lds.size.19.align.16 = internal unnamed_addr addrspace(3) global [19 x i8] undef, align 16
@k5.lds.size.20.align.16 = internal unnamed_addr addrspace(3) global [20 x i8] undef, align 16

define amdgpu_kernel void @k5() {
   store i8 1, ptr addrspace(3) @k5.lds.size.17.align.16, align 16

   store i8 2, ptr addrspace(3) @k5.lds.size.18.align.16, align 16

   store i8 3, ptr addrspace(3) @k5.lds.size.19.align.16, align 16

   store i8 4, ptr addrspace(3) @k5.lds.size.20.align.16, align 16

  ret void
}
