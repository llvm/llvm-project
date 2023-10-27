; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=load-store-vectorizer -S -o - %s | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5"

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
