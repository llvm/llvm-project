; RUN: llvm-split -o %t %s -j 4 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t3 | FileCheck --check-prefix=CHECK3 --implicit-check-not=define %s

; CHECK0: define internal void @PrivateHelper1()
; CHECK0: define amdgpu_kernel void @D

; CHECK1: define internal void @PrivateHelper0()
; CHECK1: define amdgpu_kernel void @C

; CHECK2: define internal void @OverridableHelper1()
; CHECK2: define amdgpu_kernel void @B

; CHECK3: define available_externally void @OverridableHelper0()
; CHECK3: define amdgpu_kernel void @A

define available_externally void @OverridableHelper0() {
  ret void
}

define internal void @OverridableHelper1() #0 {
  ret void
}

define internal void @PrivateHelper0() {
  ret void
}

define internal void @PrivateHelper1() {
  ret void
}

define amdgpu_kernel void @A() {
  call void @OverridableHelper0()
  ret void
}

define amdgpu_kernel void @B() {
  call void @OverridableHelper1()
  ret void
}

define amdgpu_kernel void @C() {
  call void @PrivateHelper0()
  ret void
}

define amdgpu_kernel void @D() {
  call void @PrivateHelper1()
  ret void
}

attributes #0 = { nobuiltin }
