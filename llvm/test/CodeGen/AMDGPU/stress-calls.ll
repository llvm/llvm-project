; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-stress-function-calls -amdgpu-always-inline %s | FileCheck %s

; CHECK: define internal fastcc i32 @alwaysinline_func(i32 %a) #0 {
define internal fastcc i32 @alwaysinline_func(i32 %a) alwaysinline {
entry:
  %tmp0 = add i32 %a, 1
  ret i32 %tmp0
}

; CHECK: define internal fastcc i32 @noinline_func(i32 %a) #1 {
define internal fastcc i32 @noinline_func(i32 %a) noinline {
entry:
  %tmp0 = add i32 %a, 2
  ret i32 %tmp0
}

; CHECK: define internal fastcc i32 @unmarked_func(i32 %a) #1 {
define internal fastcc i32 @unmarked_func(i32 %a) {
entry:
  %tmp0 = add i32 %a, 3
  ret i32 %tmp0
}

define amdgpu_kernel void @kernel(ptr addrspace(1) %out) {
entry:
  %tmp0 = call i32 @alwaysinline_func(i32 1)
  store volatile i32 %tmp0, ptr addrspace(1) %out
  %tmp1 = call i32 @noinline_func(i32 1)
  store volatile i32 %tmp1, ptr addrspace(1) %out
  %tmp2 = call i32 @unmarked_func(i32 1)
  store volatile i32 %tmp2, ptr addrspace(1) %out
  ret void
}

; CHECK: attributes #0 = { alwaysinline }
; CHECK: attributes #1 = { noinline }
