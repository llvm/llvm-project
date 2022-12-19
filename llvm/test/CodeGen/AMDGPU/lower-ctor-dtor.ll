; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-ctor-dtor < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-ctor-dtor < %s | FileCheck %s

; Make sure we get the same result if we run multiple times
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-ctor-dtor,amdgpu-lower-ctor-dtor < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readelf -s - 2>&1 | FileCheck %s -check-prefix=VISIBILITY

@llvm.global_ctors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @foo, ptr null }]
@llvm.global_dtors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @bar, ptr null }]

; CHECK-NOT: @llvm.global_ctors
; CHECK-NOT: @llvm.global_dtors

; CHECK-LABEL: amdgpu_kernel void @amdgcn.device.init() #0
; CHECK-NEXT: call void @foo
; CHECK-NEXT: ret void

; CHECK-LABEL: amdgpu_kernel void @amdgcn.device.fini() #1
; CHECK-NEXT: call void @bar
; CHECK-NEXT: ret void

; CHECK-NOT: amdgcn.device.

; VISIBILITY: FUNC   GLOBAL PROTECTED {{.*}} amdgcn.device.init
; VISIBILITY: OBJECT GLOBAL DEFAULT {{.*}} amdgcn.device.init.kd
; VISIBILITY: FUNC   GLOBAL PROTECTED {{.*}} amdgcn.device.fini
; VISIBILITY: OBJECT   GLOBAL DEFAULT {{.*}} amdgcn.device.fini.kd

define internal void @foo() {
  ret void
}

define internal void @bar() {
  ret void
}

; CHECK: attributes #0 = { "device-init" }
; CHECK: attributes #1 = { "device-fini" }
