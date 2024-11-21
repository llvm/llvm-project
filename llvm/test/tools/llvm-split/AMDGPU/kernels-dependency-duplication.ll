; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 %s

; 3 kernels share a common helper, that helper should be
; cloned in all partitions.

; CHECK0-NOT: define
; CHECK0: define internal void @Helper
; CHECK0: define amdgpu_kernel void @C
; CHECK0-NOT: define

; CHECK1-NOT: define
; CHECK1: define internal void @Helper
; CHECK1: define amdgpu_kernel void @B
; CHECK1-NOT: define

; CHECK2-NOT: define
; CHECK2: define internal void @Helper
; CHECK2: define amdgpu_kernel void @A
; CHECK2-NOT: define

define internal void @Helper() {
  ret void
}

define amdgpu_kernel void @A() {
  call void @Helper()
  ret void
}

define amdgpu_kernel void @B() {
  call void @Helper()
  ret void
}

define amdgpu_kernel void @C() {
  call void @Helper()
  ret void
}
