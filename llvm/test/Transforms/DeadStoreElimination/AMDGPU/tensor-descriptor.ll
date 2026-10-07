; NOTE: Do not autogenerate. Checks ensure descriptor-based DMA observes ordinary memory.
; RUN: opt -passes=dse -S < %s | FileCheck %s

target triple = "amdgpu12.50-amd-amdhsa"

; The descriptor can name the same global allocation as %global, even though
; its address is passed in integer vector lanes rather than as a pointer.
define void @tensor_load(<4 x i32> inreg %d0, <8 x i32> inreg %d1, ptr addrspace(1) %global) {
; CHECK-LABEL: define void @tensor_load(
; CHECK: store i32 1, ptr addrspace(1) %global
; CHECK: call void @llvm.amdgcn.tensor.load.to.lds(
; CHECK: call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
; CHECK: store i32 2, ptr addrspace(1) %global
  store i32 1, ptr addrspace(1) %global
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %d0, <8 x i32> %d1, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  store i32 2, ptr addrspace(1) %global
  ret void
}

; A tensor store can read the same LDS allocation named by %lds.
define void @tensor_store(<4 x i32> inreg %d0, <8 x i32> inreg %d1, ptr addrspace(3) %lds) {
; CHECK-LABEL: define void @tensor_store(
; CHECK: store i32 1, ptr addrspace(3) %lds
; CHECK: call void @llvm.amdgcn.tensor.store.from.lds(
; CHECK: call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
; CHECK: store i32 2, ptr addrspace(3) %lds
  store i32 1, ptr addrspace(3) %lds
  call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %d0, <8 x i32> %d1, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <8 x i32> zeroinitializer, i32 0)
  call void @llvm.amdgcn.s.wait.tensorcnt(i16 0)
  store i32 2, ptr addrspace(3) %lds
  ret void
}
