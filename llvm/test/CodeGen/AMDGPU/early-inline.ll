; RUN: opt -mtriple=amdgcn-- -O1 -S -inline-threshold=1 -amdgpu-early-inline-all %s | FileCheck %s

@c_alias = dso_local alias i32 (i32), ptr @callee

define dso_local i32 @callee(i32 %x) {
entry:
  %mul1 = mul i32 %x, %x
  %mul2 = mul i32 %mul1, %x
  %mul3 = mul i32 %mul1, %mul2
  %mul4 = mul i32 %mul3, %mul2
  %mul5 = mul i32 %mul4, %mul3
  ret i32 %mul5
}

; CHECK-LABEL: @caller
; CHECK: mul i32
; CHECK-NOT: call i32

define amdgpu_kernel void @caller(i32 %x) {
entry:
  %res = call i32 @callee(i32 %x)
  store volatile i32 %res, ptr addrspace(1) undef
  ret void
}

; CHECK-LABEL: @alias_caller(
; CHECK-NOT: call
; CHECK: {{^[}]}}
define amdgpu_kernel void @alias_caller(i32 %x) {
entry:
  %res = call i32 @c_alias(i32 %x)
  store volatile i32 %res, ptr addrspace(1) undef
  ret void
}
