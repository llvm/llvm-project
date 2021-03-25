; RUN: opt -S -mtriple=amdgcn-- -amdgpu-lower-module-lds < %s | FileCheck %s
; RUN: opt -S -mtriple=amdgcn-- -passes=amdgpu-lower-module-lds < %s | FileCheck %s

; Padding to meet alignment, so references to @var1 replaced with gep ptr, 0, 2
; No i64 as addrspace(3) types with initializers are ignored. Likewise no addrspace(4).
; CHECK: %llvm.amdgcn.module.lds.t = type { float, [4 x i8], i32 }

; Variables removed by pass
; CHECK-NOT: @var0
; CHECK-NOT: @var1

@var0 = addrspace(3) global float undef, align 8
@var1 = addrspace(3) global i32 undef, align 8

@ptr =  addrspace(1) global i32 addrspace(3)* @var1, align 4

; A variable that is unchanged by pass
; CHECK: @with_init = addrspace(3) global i64 0
@with_init = addrspace(3) global i64 0

; Instance of new type, aligned to max of element alignment
; CHECK: @llvm.amdgcn.module.lds = internal addrspace(3) global %llvm.amdgcn.module.lds.t undef, align 8

; Use in func rewritten to access struct at address zero, which prints as null
; CHECK-LABEL: @func()
; CHECK: %dec = atomicrmw fsub float addrspace(3)* null, float 1.0
; CHECK: %val0 = load i32, i32 addrspace(3)* getelementptr (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* null, i32 0, i32 2), align 4
; CHECK: %val1 = add i32 %val0, 4
; CHECK: store i32 %val1, i32 addrspace(3)* getelementptr (%llvm.amdgcn.module.lds.t, %llvm.amdgcn.module.lds.t addrspace(3)* null, i32 0, i32 2), align 4
; CHECK: %unused0 = atomicrmw add i64 addrspace(3)* @with_init, i64 1 monotonic
define void @func() {
  %dec = atomicrmw fsub float addrspace(3)* @var0, float 1.0 monotonic
  %val0 = load i32, i32 addrspace(3)* @var1, align 4
  %val1 = add i32 %val0, 4
  store i32 %val1, i32 addrspace(3)* @var1, align 4
  %unused0 = atomicrmw add i64 addrspace(3)* @with_init, i64 1 monotonic
  ret void
}

; This kernel calls a function that uses LDS so needs the block
; CHECK-LABEL: @kern_call()
; CHECK: call void @llvm.donothing() [ "ExplicitUse"(%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds) ]
; CHECK: call void @func()
; CHECK: %dec = atomicrmw fsub float addrspace(3)* null, float 2.0
define amdgpu_kernel void @kern_call() {
  call void @func()
  %dec = atomicrmw fsub float addrspace(3)* @var0, float 2.0 monotonic
  ret void
}

; This kernel does not need to alloc the LDS block as it makes no calls
; CHECK-LABEL: @kern_empty()
; CHECK: call void @llvm.donothing() [ "ExplicitUse"(%llvm.amdgcn.module.lds.t addrspace(3)* @llvm.amdgcn.module.lds) ]
define spir_kernel void @kern_empty() {
  ret void
}
