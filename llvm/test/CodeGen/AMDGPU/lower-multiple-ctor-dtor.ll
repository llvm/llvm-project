; RUN: opt -S -mtriple=amdgcn--  -amdgpu-lower-ctor-dtor < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readelf -s - 2>&1 | FileCheck %s -check-prefix=CHECK-VIS

@llvm.global_ctors = appending addrspace(1) global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* @foo, i8* null }, { i32, void ()*, i8* } { i32 1, void ()* @foo.5, i8* null }]
@llvm.global_dtors = appending addrspace(1) global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* @bar, i8* null }, { i32, void ()*, i8* } { i32 1, void ()* @bar.5, i8* null }]

; CHECK-LABEL: amdgpu_kernel void @amdgcn.device.init() #0
; CHECK-NEXT: call void @foo
; CHECK-NEXT: call void @foo.5

; CHECK-LABEL: amdgpu_kernel void @amdgcn.device.fini() #1
; CHECK-NEXT: call void @bar
; CHECK-NEXT: call void @bar.5

; CHECK-VIS: FUNC   GLOBAL PROTECTED {{.*}} amdgcn.device.init
; CHECK-VIS: OBJECT GLOBAL DEFAULT {{.*}} amdgcn.device.init.kd
; CHECK-VIS: FUNC   GLOBAL PROTECTED {{.*}} amdgcn.device.fini
; CHECK-VIS: OBJECT   GLOBAL DEFAULT {{.*}} amdgcn.device.fini.kd

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