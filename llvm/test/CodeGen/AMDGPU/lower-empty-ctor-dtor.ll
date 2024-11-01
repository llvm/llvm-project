; RUN: opt -S -mtriple=amdgcn--  -amdgpu-lower-ctor-dtor < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readelf -s - 2>&1 | FileCheck %s

@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer
@llvm.global_dtors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

; No amdgpu_kernels emitted for empty global_ctors
; CHECK-NOT: amdgcn.device.init
; CHECK-NOT: amdgcn.device.fini
