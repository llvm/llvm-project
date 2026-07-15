; RUN: llc -mtriple=amdgpu7.00-amd-amdhsa < %s | FileCheck %s

@llvm.global_ctors = external global [2 x { i32, ptr, ptr }]
@llvm.global_dtors = external global [2 x { i32, ptr, ptr }]

; No amdgpu_kernels emitted for global_ctors declaration
; CHECK-NOT: amdgcn.device.init
; CHECK-NOT: amdgcn.device.fini
