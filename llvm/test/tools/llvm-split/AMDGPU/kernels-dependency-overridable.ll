; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 %s

; CHECK0-NOT: define
; CHECK0: define void @ExternalHelper
; CHECK0: define amdgpu_kernel void @A
; CHECK0: define amdgpu_kernel void @B
; CHECK0-NOT: define

; CHECK1-NOT: define
; CHECK1: define amdgpu_kernel void @D
; CHECK1-NOT: define

; CHECK2-NOT: define
; CHECK2: define amdgpu_kernel void @C
; CHECK2-NOT: define

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
