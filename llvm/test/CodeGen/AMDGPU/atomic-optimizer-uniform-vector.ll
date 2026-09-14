; RUN: opt -S -mtriple=amdgpu9.00-amd-amdhsa -passes='amdgpu-atomic-optimizer<strategy=dpp>,verify<domtree>' -verify-each %s | FileCheck %s
; RUN: opt -S -mtriple=amdgpu9.00-amd-amdhsa -passes='amdgpu-atomic-optimizer<strategy=iterative>,verify<domtree>' -verify-each %s | FileCheck %s
; RUN: opt -S -mtriple=amdgpu12.01-amd-amdhsa -passes='amdgpu-atomic-optimizer<strategy=dpp>,verify<domtree>' -verify-each %s | FileCheck %s
; RUN: opt -S -mtriple=amdgpu12.01-amd-amdhsa -passes='amdgpu-atomic-optimizer<strategy=iterative>,verify<domtree>' -verify-each %s | FileCheck %s

; Both operands are uniform. The scalar lane count must not be cast to a vector.
; CHECK-LABEL: define amdgpu_kernel void @global_uniform_vector_add(
; CHECK-NEXT: entry:
; CHECK-NEXT: %old = atomicrmw add ptr addrspace(1) %ptr, <2 x i32> <i32 1, i32 2> monotonic, align 8
; CHECK-NEXT: ret void

define amdgpu_kernel void @global_uniform_vector_add(ptr addrspace(1) %ptr) {
entry:
  %old = atomicrmw add ptr addrspace(1) %ptr, <2 x i32> <i32 1, i32 2> monotonic
  ret void
}

; Cover result reconstruction for an idempotent operation and the LDS path.
; CHECK-LABEL: define amdgpu_kernel void @local_uniform_vector_and(
; CHECK-NEXT: entry:
; CHECK-NEXT: %old = atomicrmw and ptr addrspace(3) %ptr, <2 x i32> <i32 1, i32 2> monotonic, align 8
; CHECK-NEXT: store <2 x i32> %old, ptr addrspace(1) %out, align 8
; CHECK-NEXT: ret void

define amdgpu_kernel void @local_uniform_vector_and(ptr addrspace(3) %ptr, ptr addrspace(1) %out) {
entry:
  %old = atomicrmw and ptr addrspace(3) %ptr, <2 x i32> <i32 1, i32 2> monotonic
  store <2 x i32> %old, ptr addrspace(1) %out
  ret void
}

; Scalar uniform operands narrower than 32 bits remain eligible. Moving the
; divergent-value isLegalCrossLaneType check to the common path would lose this.
; CHECK-LABEL: define amdgpu_kernel void @global_uniform_scalar_i16(
; CHECK: call {{.*}} @llvm.amdgcn.ballot.
; CHECK: atomicrmw add ptr addrspace(1) %ptr, i16 %{{.*}} monotonic

define amdgpu_kernel void @global_uniform_scalar_i16(ptr addrspace(1) %ptr, i16 %val) {
entry:
  %old = atomicrmw add ptr addrspace(1) %ptr, i16 %val monotonic
  ret void
}
