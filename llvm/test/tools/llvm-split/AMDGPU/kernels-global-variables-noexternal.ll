; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-no-externalize-globals
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 --implicit-check-not=define %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 --implicit-check-not=define %s

; 3 kernels use private/internal global variables.
; The GVs should be copied in each partition as needed.

; CHECK0: @bar = internal constant ptr
; CHECK0: define amdgpu_kernel void @C

; CHECK1: @foo = private constant ptr
; CHECK1: define amdgpu_kernel void @A

; CHECK2: @foo = private constant ptr
; CHECK2: @bar = internal constant ptr
; CHECK2: define amdgpu_kernel void @B

@foo = private constant ptr poison
@bar = internal constant ptr poison

define amdgpu_kernel void @A() {
  store i32 42, ptr @foo
  ret void
}

define amdgpu_kernel void @B() {
  store i32 42, ptr @foo
  store i32 42, ptr @bar
  ret void
}

define amdgpu_kernel void @C() {
  store i32 42, ptr @bar
  ret void
}
