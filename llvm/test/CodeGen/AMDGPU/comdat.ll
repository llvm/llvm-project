; RUN: llc -mtriple amdgcn-amd-amdhsa -filetype=obj <%s \
; RUN:   | llvm-readobj -symbols - | FileCheck %s

; CHECK: Name: func1
; CHECK: Section: .text.func1

; CHECK: Name: func2
; CHECK: Section: .text.func2

$func1 = comdat any
$func2 = comdat any

define amdgpu_kernel void @func1() local_unnamed_addr comdat {
  ret void
}

define amdgpu_kernel void @func2() local_unnamed_addr comdat {
  ret void
}
