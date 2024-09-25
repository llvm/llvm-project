; RUN: llvm-split -o %t %s -j 2 -mtriple amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=define %s

; 3 kernels:
;   - A calls nothing
;   - B calls @PerryThePlatypus
;   - C calls @Perry, an alias of @PerryThePlatypus
;
; We should see through the alias and put B/C in the same
; partition.
;
; Additionally, @PerryThePlatypus gets externalized as
; the alias counts as taking its address.

; CHECK0: define amdgpu_kernel void @A

; CHECK1: @Perry = internal alias ptr (), ptr @PerryThePlatypus
; CHECK1: define hidden void @PerryThePlatypus()
; CHECK1: define amdgpu_kernel void @B
; CHECK1: define amdgpu_kernel void @C

@Perry = internal alias ptr(), ptr @PerryThePlatypus

define internal void @PerryThePlatypus() {
  ret void
}

define amdgpu_kernel void @A() {
  ret void
}

define amdgpu_kernel void @B() {
  call void @PerryThePlatypus()
  ret void
}

define amdgpu_kernel void @C() {
  call void @Perry()
  ret void
}
