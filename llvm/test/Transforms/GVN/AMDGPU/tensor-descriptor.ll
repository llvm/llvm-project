; NOTE: Do not autogenerate. Checks ensure tensor DMA writes invalidate ordinary loads.
; RUN: opt -passes=gvn -S < %s | FileCheck %s

target triple = "amdgpu12.50-amd-amdhsa"

; A tensor load can write the same LDS allocation named by %dst.
define i32 @tensor_load_writes(<4 x i32> inreg %d0, <8 x i32> inreg %d1, ptr addrspace(3) %dst) {
; CHECK-LABEL: define i32 @tensor_load_writes(
; CHECK: %before = load i32, ptr addrspace(3) %dst
; CHECK: call void @llvm.amdgcn.tensor.load.to.lds(
; CHECK: call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
; CHECK: %after = load i32, ptr addrspace(3) %dst
; CHECK: %sum = add i32 %before, %after
  %before = load i32, ptr addrspace(3) %dst
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %d0, <8 x i32> %d1, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  %after = load i32, ptr addrspace(3) %dst
  %sum = add i32 %before, %after
  ret i32 %sum
}

; A tensor store can write the same global allocation named by %dst.
define i32 @tensor_store_writes(<4 x i32> inreg %d0, <8 x i32> inreg %d1, ptr addrspace(1) %dst) {
; CHECK-LABEL: define i32 @tensor_store_writes(
; CHECK: %before = load i32, ptr addrspace(1) %dst
; CHECK: call void @llvm.amdgcn.tensor.store.from.lds(
; CHECK: call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
; CHECK: %after = load i32, ptr addrspace(1) %dst
; CHECK: %sum = add i32 %before, %after
  %before = load i32, ptr addrspace(1) %dst
  call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %d0, <8 x i32> %d1, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  %after = load i32, ptr addrspace(1) %dst
  %sum = add i32 %before, %after
  ret i32 %sum
}
