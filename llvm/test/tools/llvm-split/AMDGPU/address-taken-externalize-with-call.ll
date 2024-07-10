; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-large-function-threshold=0
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 %s

; 3 kernels:
;   - A does a direct call to HelperA
;   - B is storing @HelperA
;   - C does a direct call to HelperA
;
; The helper functions will get externalized, which will force A and C into P0 as
; external functions cannot be duplicated.

; CHECK0: define hidden void @HelperA()
; CHECK0: define amdgpu_kernel void @A()
; CHECK0: declare amdgpu_kernel void @B(ptr)
; CHECK0: define amdgpu_kernel void @C()

; CHECK1: declare hidden void @HelperA()
; CHECK1: declare amdgpu_kernel void @A()
; CHECK1: declare amdgpu_kernel void @B(ptr)
; CHECK1: declare amdgpu_kernel void @C()

; CHECK2: declare hidden void @HelperA()
; CHECK2: declare amdgpu_kernel void @A()
; CHECK2: define amdgpu_kernel void @B(ptr %dst)
; CHECK2: declare amdgpu_kernel void @C()

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
