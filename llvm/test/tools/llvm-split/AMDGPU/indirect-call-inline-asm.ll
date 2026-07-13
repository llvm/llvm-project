; RUN: llvm-split -o %t %s -j 2 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-no-externalize-address-taken
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=define %s

; CHECK0: define internal void @HelperB
; CHECK0: define amdgpu_kernel void @B

; CHECK1: define internal void @HelperA()
; CHECK1: define amdgpu_kernel void @A()

@addrthief = global [2 x ptr] [ptr @HelperA, ptr @HelperB]

define internal void @HelperA() {
  ret void
}

define internal void @HelperB() {
  ret void
}

define amdgpu_kernel void @A() {
  call void asm sideeffect "v_mov_b32 v0, 7", "~{v0}"()
  call void @HelperA()
  ret void
}

define amdgpu_kernel void @B(ptr %out) {
  call void @HelperB()
  ret void
}
