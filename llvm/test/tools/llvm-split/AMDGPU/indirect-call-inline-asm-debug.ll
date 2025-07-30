; REQUIRES: asserts

; RUN: llvm-split -o %t %s -j 2 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-no-externalize-address-taken -debug-only=amdgpu-split-module 2>&1 | FileCheck %s

; CHECK:      [!] callgraph is incomplete for ptr @A  - analyzing function
; CHECK-NEXT:     found inline assembly
; CHECK-NOT:      indirect call found

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
