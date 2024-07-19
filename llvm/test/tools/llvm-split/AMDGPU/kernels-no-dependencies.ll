; RUN: llvm-split -o %t %s -j 4 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 %s
; RUN: llvm-dis -o - %t3 | FileCheck --check-prefix=CHECK3 %s

; Check that 4 independent kernels get put into 4 different partitions.

; CHECK0-NOT: define
; CHECK0: define amdgpu_kernel void @D
; CHECK0-NOT: define

; CHECK1-NOT: define
; CHECK1: define amdgpu_kernel void @C
; CHECK1-NOT: define

; CHECK2-NOT: define
; CHECK2: define amdgpu_kernel void @B
; CHECK2-NOT: define

; CHECK3-NOT: define
; CHECK3: define amdgpu_kernel void @A
; CHECK3-NOT: define

define amdgpu_kernel void @A() {
  ret void
}

define amdgpu_kernel void @B() {
  ret void
}

define amdgpu_kernel void @C() {
  ret void
}

define amdgpu_kernel void @D() {
  ret void
}
