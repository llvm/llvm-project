; RUN: opt -mtriple=amdgcn-amd-amdhsa -load-store-vectorizer -S -o - %s | FileCheck %s

; CHECK-LABEL: @optnone(
; CHECK: store i32
; CHECK: store i32
define amdgpu_kernel void @optnone(i32 addrspace(1)* %out) noinline optnone {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1

  store i32 123, i32 addrspace(1)* %out.gep.1
  store i32 456, i32 addrspace(1)* %out
  ret void
}

; CHECK-LABEL: @do_opt(
; CHECK: store <2 x i32>
define amdgpu_kernel void @do_opt(i32 addrspace(1)* %out) {
  %out.gep.1 = getelementptr i32, i32 addrspace(1)* %out, i32 1

  store i32 123, i32 addrspace(1)* %out.gep.1
  store i32 456, i32 addrspace(1)* %out
  ret void
}
