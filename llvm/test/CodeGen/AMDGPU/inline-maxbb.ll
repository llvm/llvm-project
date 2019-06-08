; RUN: opt -mtriple=amdgcn-- --amdgpu-inline -S -amdgpu-inline-max-bb=2 %s | FileCheck %s --check-prefix=NOINL
; RUN: opt -mtriple=amdgcn-- --amdgpu-inline -S -amdgpu-inline-max-bb=3 %s | FileCheck %s --check-prefix=INL

define i32 @callee(i32 %x) {
entry:
  %cc = icmp eq i32 %x, 1
  br i1 %cc, label %ret_res, label %mulx

mulx:
  %mul1 = mul i32 %x, %x
  %mul2 = mul i32 %mul1, %x
  %mul3 = mul i32 %mul1, %mul2
  %mul4 = mul i32 %mul3, %mul2
  %mul5 = mul i32 %mul4, %mul3
  br label %ret_res

ret_res:
  %r = phi i32 [ %mul5, %mulx ], [ %x, %entry ]
  ret i32 %r
}

; INL-LABEL: @caller
; NOINL-LABEL: @caller
; INL: mul i32
; INL-NOT: call i32
; NOINL-NOT: mul i32
; NOINL: call i32

define amdgpu_kernel void @caller(i32 %x) {
  %res = call i32 @callee(i32 %x)
  store volatile i32 %res, i32 addrspace(1)* undef
  ret void
}
