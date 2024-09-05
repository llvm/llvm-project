; RUN: llvm-split -o %t %s -j 4 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 %s
; RUN: llvm-dis -o - %t3 | FileCheck --check-prefix=CHECK3 %s

; Both overridable helper should go in P0.

; CHECK0-NOT: define
; CHECK0: define available_externally void @OverridableHelper0()
; CHECK0: define internal void @OverridableHelper1()
; CHECK0: define amdgpu_kernel void @A
; CHECK0: define amdgpu_kernel void @B
; CHECK0-NOT: define

; CHECK1-NOT: define

; CHECK2-NOT: define
; CHECK2: define internal void @PrivateHelper1()
; CHECK2: define amdgpu_kernel void @D
; CHECK2-NOT: define

; CHECK3-NOT: define
; CHECK3: define internal void @PrivateHelper0()
; CHECK3: define amdgpu_kernel void @C
; CHECK3-NOT: define

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
