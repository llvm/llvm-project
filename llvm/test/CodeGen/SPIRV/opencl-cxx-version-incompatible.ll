; RUN: not --crash llc -O0 -mtriple=spirv64-unknown-unknown %s -o - 2>&1 | FileCheck %s

; Incompatible OpenCL C and C++ versions should produce a fatal error.
; OpenCL C 2.0 is not compatible with C++ for OpenCL 2021.

; CHECK: LLVM ERROR: opencl cxx version is not compatible with opencl c version!

define spir_kernel void @foo() {
entry:
  ret void
}

!opencl.ocl.version = !{!0}
!opencl.cxx.version = !{!1}

!0 = !{i32 2, i32 0}
!1 = !{i32 2021, i32 0}
