; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 < %s | FileCheck %s

@llvm.global_ctors = external global [2 x { i32, void ()*, i8* }]
@llvm.global_dtors = external global [2 x { i32, void ()*, i8* }]

; No amdgpu_kernels emitted for global_ctors declaration
; CHECK-NOT: amdgcn.device.init
; CHECK-NOT: amdgcn.device.fini
