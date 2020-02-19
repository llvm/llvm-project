; RUN: not --crash llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck %s

; FIXME: It should be invalid IR to have a call to a kernel, but this
; is currently relied on, but should be eliminated before codegen.
define amdgpu_kernel void @callee_kernel(i32 addrspace(1)* %out) #0 {
entry:
  store volatile i32 0, i32 addrspace(1)* %out
  ret void
}

; CHECK: LLVM ERROR: Unsupported calling convention for call
define amdgpu_kernel void @caller_kernel(i32 addrspace(1)* %out) #0 {
entry:
  call amdgpu_kernel void @callee_kernel(i32 addrspace(1)* %out)
  ret void
}

attributes #0 = { nounwind noinline }
