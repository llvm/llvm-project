; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds --amdgpu-lower-module-lds-strategy=module < %s | FileCheck %s

; Padding to meet alignment, so references to @var1 replaced with gep ptr, 0, 2
; No i64 as addrspace(3) types with initializers are ignored. Likewise no addrspace(4).
; CHECK: %llvm.amdgcn.module.lds.t = type { float, [4 x i8], i32 }

; Variable removed by pass
; CHECK-NOT: @var0

@var0 = addrspace(3) global float undef, align 8
@var1 = addrspace(3) global i32 undef, align 8

; The invalid use by the global is left unchanged
; CHECK: @var1 = addrspace(3) global i32 undef, align 8
; CHECK: @ptr = addrspace(1) global ptr addrspace(3) @var1, align 4
@ptr = addrspace(1) global ptr addrspace(3) @var1, align 4

; A variable that is unchanged by pass
; CHECK: @with_init = addrspace(3) global i64 0
@with_init = addrspace(3) global i64 0

; Instance of new type, aligned to max of element alignment
; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t undef, align 8

; Use in func rewritten to access struct at address zero
; CHECK-LABEL: @func()
; CHECK: %dec = atomicrmw fsub ptr addrspace(3) @llvm.amdgcn.module.lds, float 1.0
; CHECK: %val0 = load i32, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.module.lds.t, ptr addrspace(3) @llvm.amdgcn.module.lds, i32 0, i32 2), align 8
; CHECK: %val1 = add i32 %val0, 4
; CHECK: store i32 %val1, ptr addrspace(3) getelementptr inbounds (%llvm.amdgcn.module.lds.t, ptr addrspace(3) @llvm.amdgcn.module.lds, i32 0, i32 2), align 8
; CHECK: %unused0 = atomicrmw add ptr addrspace(3) @with_init, i64 1 monotonic
define void @func() {
  %dec = atomicrmw fsub ptr addrspace(3) @var0, float 1.0 monotonic
  %val0 = load i32, ptr addrspace(3) @var1, align 4
  %val1 = add i32 %val0, 4
  store i32 %val1, ptr addrspace(3) @var1, align 4
  %unused0 = atomicrmw add ptr addrspace(3) @with_init, i64 1 monotonic
  ret void
}

; This kernel calls a function that uses LDS so needs the block
; CHECK-LABEL: @kern_call() #0
; CHECK: call void @llvm.donothing() [ "ExplicitUse"(ptr addrspace(3) @llvm.amdgcn.module.lds) ]
; CHECK: call void @func()
; CHECK: %dec = atomicrmw fsub ptr addrspace(3) @llvm.amdgcn.module.lds, float 2.000000e+00 monotonic, align 8
define amdgpu_kernel void @kern_call() {
  call void @func()
  %dec = atomicrmw fsub ptr addrspace(3) @var0, float 2.0 monotonic
  ret void
}

; This kernel does alloc the LDS block as it makes no calls
; CHECK-LABEL: @kern_empty()
; CHECK-NOT: call void @llvm.donothing()
define spir_kernel void @kern_empty() {
  ret void
}

; Make sure we don't crash trying to insert code into a kernel
; declaration.
declare amdgpu_kernel void @kernel_declaration()

; CHECK: attributes #0 = { "amdgpu-lds-size"="12" }
