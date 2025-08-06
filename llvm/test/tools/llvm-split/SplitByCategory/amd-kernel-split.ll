; -- Per-kernel split
; RUN: llvm-split -split-by-category=kernel -S < %s -o %tC
; RUN: FileCheck %s -input-file=%tC_0.ll --check-prefixes CHECK-A0
; RUN: FileCheck %s -input-file=%tC_1.ll --check-prefixes CHECK-A1

define dso_local amdgpu_kernel void @KernelA() {
  ret void
}

define dso_local amdgpu_kernel void @KernelB() {
  ret void
}

; CHECK-A0: define dso_local amdgpu_kernel void @KernelB()
; CHECK-A0-NOT: define dso_local amdgpu_kernel void @KernelA()
; CHECK-A1-NOT: define dso_local amdgpu_kernel void @KernelB()
; CHECK-A1: define dso_local amdgpu_kernel void @KernelA()
