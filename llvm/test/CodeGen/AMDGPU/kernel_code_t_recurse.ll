; RUN: llc -mtriple=amdgcn-mesa-mesa3d < %s | FileCheck %s

; CHECK-LABEL: non_kernel_recursion:
; CHECK: .set non_kernel_recursion.has_recursion, 1
; CHECK: .set non_kernel_recursion.has_indirect_call, 0
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
; CHECK: is_dynamic_callstack = kernel_caller_recursion.has_dyn_sized_stack|kernel_caller_recursion.has_recursion
; CHECK: .end_amd_kernel_code_t

; CHECK: .set kernel_caller_recursion.has_recursion, or(1, non_kernel_recursion.has_recursion)
; CHECK: .set kernel_caller_recursion.has_indirect_call, or(0, non_kernel_recursion.has_indirect_call)
define amdgpu_kernel void @kernel_caller_recursion(i32 %n) #0 {
  call void @non_kernel_recursion(i32 %n)
  ret void
}
