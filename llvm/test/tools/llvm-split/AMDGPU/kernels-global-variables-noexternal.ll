; RUN: llvm-split -o %t %s -j 3 -mtriple amdgcn-amd-amdhsa -amdgpu-module-splitting-no-externalize-globals
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 %s

; 3 kernels use private/internal global variables.
; The GVs should be copied in each partition as needed.

; CHECK0-NOT: define
; CHECK0: @bar = internal constant ptr
; CHECK0: define amdgpu_kernel void @C
; CHECK0-NOT: define

; CHECK1-NOT: define
; CHECK1: @foo = private constant ptr
; CHECK1: define amdgpu_kernel void @A
; CHECK1-NOT: define

; CHECK2-NOT: define
; CHECK2: @foo = private constant ptr
; CHECK2: @bar = internal constant ptr
; CHECK2: define amdgpu_kernel void @B
; CHECK2-NOT: define

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
