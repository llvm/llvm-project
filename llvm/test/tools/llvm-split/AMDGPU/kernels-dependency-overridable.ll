; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 --implicit-check-not=define %s

; CHECK0: define amdgpu_kernel void @D

; CHECK1: define amdgpu_kernel void @C

; CHECK2: define void @ExternalHelper
; CHECK2: define amdgpu_kernel void @A
; CHECK2: define amdgpu_kernel void @B

define void @ExternalHelper() {
  ret void
}

define amdgpu_kernel void @A() {
  call void @ExternalHelper()
  ret void
}

define amdgpu_kernel void @B() {
  call void @ExternalHelper()
  ret void
}

define amdgpu_kernel void @C() {
  ret void
}

define amdgpu_kernel void @D() {
  ret void
}
