; RUN: not llc -mtriple=amdgcn-amd- -mcpu=gfx803 < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s 2>&1 | FileCheck %s

; CHECK: error: <unknown>:0:0: in function invalid_fence void (): Unsupported atomic synchronization scope
define amdgpu_kernel void @invalid_fence() {
entry:
  fence syncscope("invalid") seq_cst
  ret void
}

; CHECK: error: <unknown>:0:0: in function invalid_load void (ptr, ptr): Unsupported non-inclusive atomic synchronization scope
define amdgpu_kernel void @invalid_load(
    ptr %in, ptr %out) {
entry:
  %val = load atomic i32, ptr %in syncscope("invalid") seq_cst, align 4
  store i32 %val, ptr %out
  ret void
}

; CHECK: error: <unknown>:0:0: in function invalid_store void (i32, ptr): Unsupported non-inclusive atomic synchronization scope
define amdgpu_kernel void @invalid_store(
    i32 %in, ptr %out) {
entry:
  store atomic i32 %in, ptr %out syncscope("invalid") seq_cst, align 4
  ret void
}

; CHECK: error: <unknown>:0:0: in function invalid_cmpxchg void (ptr, i32, i32): Unsupported non-inclusive atomic synchronization scope
define amdgpu_kernel void @invalid_cmpxchg(
    ptr %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, ptr %out, i32 4
  %val = cmpxchg volatile ptr %gep, i32 %old, i32 %in syncscope("invalid") seq_cst seq_cst
  ret void
}

; CHECK: error: <unknown>:0:0: in function invalid_rmw void (ptr, i32): Unsupported non-inclusive atomic synchronization scope
define amdgpu_kernel void @invalid_rmw(
    ptr %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg ptr %out, i32 %in syncscope("invalid") seq_cst
  ret void
}
