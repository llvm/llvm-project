; RUN: llc -mtriple=amdgcn-mesa-mesa3d < %s | FileCheck %s

; CHECK-LABEL: non_kernel_recursion:
define void @non_kernel_recursion(i32 %val) #2 {
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %ret, label %call

call:
  %val.sub1 = sub i32 %val, 1
  call void @non_kernel_recursion(i32 %val.sub1)
  br label %ret

ret:
  ret void
}

; CHECK-LABEL: kernel_caller_recursion:
; CHECK: .amd_kernel_code_t
; CHECK: is_dynamic_callstack = 1
; CHECK: .end_amd_kernel_code_t
define amdgpu_kernel void @kernel_caller_recursion(i32 %n) #0 {
  call void @non_kernel_recursion(i32 %n)
  ret void
}
