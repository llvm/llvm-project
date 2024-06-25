; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 %s

; Test load balancing logic with 6 kernels.
;
; Kernels go from most expensive (A == 6) to least expensive (F == 1)
;
; Load balancing should work like this (current partition cost is in parens)
;
; Initial    -> [P0(0), P1(0), P2(0)]
;
; A(6) goes in 2 -> [P2(6), P0(0), P1(0)]
; B(5) goes in 1 -> [P2(6), P1(5), P0(4)]
; C(4) goes in 0 -> [P2(6), P1(5), P0(4)]

; D(3) goes in 0 -> [P0(7), P2(6), P1(5)]
; E(2) goes in 1 -> [P0(7), P1(7), P2(6)]
; F(1) goes in 2 -> [P0(7), P1(7), P2(7)]

; CHECK0-NOT: define
; CHECK0: define amdgpu_kernel void @C
; CHECK0: define amdgpu_kernel void @D
; CHECK0-NOT: define

; CHECK1-NOT: define
; CHECK1: define amdgpu_kernel void @B
; CHECK1: define amdgpu_kernel void @E
; CHECK1-NOT: define

; CHECK2-NOT: define
; CHECK2: define amdgpu_kernel void @A
; CHECK2: define amdgpu_kernel void @F
; CHECK2-NOT: define


define amdgpu_kernel void @A(ptr %x) {
  store i64 42, ptr %x
  store i64 43, ptr %x
  store i64 44, ptr %x
  store i64 45, ptr %x
  store i64 46, ptr %x
  ret void
}

define amdgpu_kernel void @B(ptr %x) {
  store i64 42, ptr %x
  store i64 43, ptr %x
  store i64 44, ptr %x
  store i64 45, ptr %x
  ret void
}

define amdgpu_kernel void @C(ptr %x) {
  store i64 42, ptr %x
  store i64 43, ptr %x
  store i64 44, ptr %x
  ret void
}

define amdgpu_kernel void @D(ptr %x) {
  store i64 42, ptr %x
  store i64 43, ptr %x
  ret void
}

define amdgpu_kernel void @E(ptr %x) {
  store i64 42, ptr %x
  ret void
}

define amdgpu_kernel void @F() {
  ret void
}
