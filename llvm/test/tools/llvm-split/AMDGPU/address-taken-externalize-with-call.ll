; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-large-threshold=0
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 --implicit-check-not=define %s

; 3 kernels:
;   - A does a direct call to HelperA
;   - B is storing @HelperA
;   - C does a direct call to HelperA
;
; The helper functions will get externalized, so C/A will end up
; in the same partition.

; P0 is empty.
; CHECK0: declare

; CHECK1: define amdgpu_kernel void @B(ptr %dst)

; CHECK2: define hidden void @HelperA()
; CHECK2: define amdgpu_kernel void @A()
; CHECK2: define amdgpu_kernel void @C()

define internal void @HelperA() {
  ret void
}

define amdgpu_kernel void @A() {
  call void @HelperA()
  ret void
}

define amdgpu_kernel void @B(ptr %dst) {
  store ptr @HelperA, ptr %dst
  ret void
}

define amdgpu_kernel void @C() {
  call void @HelperA()
  ret void
}
