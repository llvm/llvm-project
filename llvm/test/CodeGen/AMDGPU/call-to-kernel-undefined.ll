; RUN: not --crash llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck %s

; FIXME: It should be invalid IR to have a call to a kernel, but this
; is currently relied on, but should be eliminated before codegen.
define amdgpu_kernel void @callee_kernel(i32 addrspace(1)* %out) #0 {
entry:
  store volatile i32 0, i32 addrspace(1)* %out
  ret void
}

; Make sure there's no crash when the callsite calling convention
; doesn't match.
; CHECK: LLVM ERROR: invalid call to entry function
define amdgpu_kernel void @caller_kernel(i32 addrspace(1)* %out) #0 {
entry:
  call void @callee_kernel(i32 addrspace(1)* %out)
  ret void
}

attributes #0 = { nounwind noinline }
