; RUN: llvm-split -o %t %s -j 2 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-large-function-threshold=0
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s

; 2 kernels:
;   - A is isolated
;   - B is storing @HelperA/B's address
;
; The helper functions should get externalized (become hidden w/ external linkage)

; CHECK0: define hidden void @HelperA()
; CHECK0: define hidden void @HelperB()
; CHECK0: define amdgpu_kernel void @A()
; CHECK0: declare amdgpu_kernel void @B(i1, ptr)

; CHECK1: declare hidden void @HelperA()
; CHECK1: declare hidden void @HelperB()
; CHECK1: declare amdgpu_kernel void @A()
; CHECK1: define amdgpu_kernel void @B(i1 %cond, ptr %dst)

define internal void @HelperA() {
  ret void
}

define internal void @HelperB() {
  ret void
}

define amdgpu_kernel void @A() {
  ret void
}

define amdgpu_kernel void @B(i1 %cond, ptr %dst) {
  %addr = select i1 %cond, ptr @HelperA, ptr @HelperB
  store ptr %addr, ptr %dst
  ret void
}
