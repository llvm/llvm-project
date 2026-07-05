; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -S -o - %s | FileCheck %s

; CHECK-LABEL: @optnone(
; CHECK: store i32
; CHECK: store i32
define amdgpu_kernel void @optnone(ptr addrspace(1) %out) noinline optnone {
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1

  store i32 123, ptr addrspace(1) %out.gep.1
  store i32 456, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: @do_opt(
; CHECK: store <2 x i32>
define amdgpu_kernel void @do_opt(ptr addrspace(1) %out) {
  %out.gep.1 = getelementptr i32, ptr addrspace(1) %out, i32 1

  store i32 123, ptr addrspace(1) %out.gep.1
  store i32 456, ptr addrspace(1) %out
  ret void
}
