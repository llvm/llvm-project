; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-ctor-dtor < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readelf -s - 2>&1 | FileCheck %s -check-prefix=CHECK-VIS

; Make sure that we don't modify the functions if amdgcn.device.init or
; amdgcn.device.fini already exit.

@llvm.global_ctors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @foo, ptr null }]
@llvm.global_dtors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @bar, ptr null }]

; CHECK-LABEL: amdgpu_kernel void @amdgcn.device.init() #0 {
; CHECK-NEXT:   store volatile i32 1, ptr addrspace(1) null
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK-LABEL: define amdgpu_kernel void @amdgcn.device.fini() #1 {
; CHECK-NEXT:    store volatile i32 0, ptr addrspace(1) null
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }

; CHECK-NOT: amdgcn.device.

; CHECK-VIS: FUNC   GLOBAL PROTECTED {{.*}} amdgcn.device.init{{$}}
; CHECK-VIS: OBJECT GLOBAL DEFAULT {{.*}} amdgcn.device.init.kd{{$}}
; CHECK-VIS: FUNC   GLOBAL PROTECTED {{.*}} amdgcn.device.fini{{$}}
; CHECK-VIS: OBJECT   GLOBAL DEFAULT {{.*}} amdgcn.device.fini.kd{{$}}

define internal void @foo() {
  ret void
}

define internal void @bar() {
  ret void
}

define amdgpu_kernel void @amdgcn.device.init() #0 {
  store volatile i32 1, ptr addrspace(1) null
  ret void
}

define amdgpu_kernel void @amdgcn.device.fini() #1 {
  store volatile i32 0, ptr addrspace(1) null
  ret void
}

attributes #0 = { "device-init" }
attributes #1 = { "device-fini" }
; CHECK: attributes #0 = { "device-init" }
; CHECK: attributes #1 = { "device-fini" }
