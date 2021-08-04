; RUN: opt -S -mtriple=amdgcn--  -amdgpu-lower-ctor-dtor < %s | FileCheck %s

@llvm.global_ctors = appending addrspace(1) global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* @foo, i8* null }, { i32, void ()*, i8* } { i32 1, void ()* @foo.5, i8* null }]
@llvm.global_dtors = appending addrspace(1) global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* @bar, i8* null }, { i32, void ()*, i8* } { i32 1, void ()* @bar.5, i8* null }]

; CHECK-LABEL: amdgpu_kernel void @amdgcn.device.init() #0
; CHECK-NEXT: call void @foo
; CHECK-NEXT: call void @foo.5

; CHECK-LABEL: amdgpu_kernel void @amdgcn.device.fini() #1
; CHECK-NEXT: call void @bar
; CHECK-NEXT: call void @bar.5

define internal void @foo() {
  ret void
}

define internal void @bar() {
  ret void
}

define internal void @foo.5() {
  ret void
}

define internal void @bar.5() {
  ret void
}

; CHECK: attributes #0 = { "device-init" }
; CHECK: attributes #1 = { "device-fini" } 
