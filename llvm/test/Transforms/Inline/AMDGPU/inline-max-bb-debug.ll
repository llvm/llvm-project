; REQUIRES: asserts
; RUN: opt -mtriple=amdgpu7.00-amd-amdhsa -passes=inline -disable-output -amdgpu-inline-max-bb=3 -debug-only=AMDGPUtti < %s 2>&1 | FileCheck %s

; CHECK: AMDGPU inline max-BB rejected inlining callee into caller: caller BBs=3, callee BBs=4, combined BBs=6, max BBs=3

define i32 @callee(i32 %x) {
entry:
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %then, label %else

then:
  br label %exit

else:
  br label %exit

exit:
  %value = phi i32 [ 1, %then ], [ 2, %else ]
  ret i32 %value
}

define amdgpu_kernel void @caller(ptr addrspace(1) %out, i32 %x) {
entry:
  %cmp = icmp slt i32 %x, 0
  br i1 %cmp, label %call, label %exit

call:
  %value = call i32 @callee(i32 %x)
  store i32 %value, ptr addrspace(1) %out, align 4
  br label %exit

exit:
  ret void
}
