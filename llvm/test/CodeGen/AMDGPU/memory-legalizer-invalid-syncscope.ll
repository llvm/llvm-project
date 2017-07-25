; RUN: not llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s 2>&1 | FileCheck %s

; CHECK: error: <unknown>:0:0: in function invalid_fence void (): Unsupported synchronization scope
define amdgpu_kernel void @invalid_fence() {
entry:
  fence syncscope("invalid") seq_cst
  ret void
}

; CHECK: error: <unknown>:0:0: in function invalid_load void (i32 addrspace(4)*, i32 addrspace(4)*): Unsupported synchronization scope
define amdgpu_kernel void @invalid_load(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("invalid") seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK: error: <unknown>:0:0: in function invalid_store void (i32, i32 addrspace(4)*): Unsupported synchronization scope
define amdgpu_kernel void @invalid_store(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("invalid") seq_cst, align 4
  ret void
}

; CHECK: error: <unknown>:0:0: in function invalid_cmpxchg void (i32 addrspace(4)*, i32, i32): Unsupported synchronization scope
define amdgpu_kernel void @invalid_cmpxchg(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("invalid") seq_cst seq_cst
  ret void
}

; CHECK: error: <unknown>:0:0: in function invalid_rmw void (i32 addrspace(4)*, i32): Unsupported synchronization scope
define amdgpu_kernel void @invalid_rmw(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("invalid") seq_cst
  ret void
}
