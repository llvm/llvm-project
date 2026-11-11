; OpLifetimeStart/OpLifetimeStop must have Size 0 when Pointer points to a
; non-void type. For typed allocas the backend should use Size 0 directly on the
; OpVariable rather than bitcasting it to uchar* to carry a byte size. In
; particular, lifetime markers alone must not introduce a uchar type or the
; Int8 capability.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED --implicit-check-not=OpBitcast
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,TYPED --implicit-check-not=OpBitcast
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s --check-prefixes=CHECK,UNTYPED --implicit-check-not=OpBitcast
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}

; Shaders drop lifetime markers entirely.
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefix=VK --implicit-check-not=OpLifetime --implicit-check-not=OpBitcast
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpCapability Int8
; CHECK-NOT: OpTypeInt 8

; VK: OpCapability Shader
; VK-NOT: OpCapability Int8
; VK-NOT: OpTypeInt 8

%struct = type { i64, float }

; CHECK-LABEL: Begin function int_var
; TYPED: %[[#IntVar:]] = OpVariable
; TYPED: OpLifetimeStart %[[#IntVar]] 0
; UNTYPED: %[[#IntVar:]] = OpUntypedVariableKHR
; UNTYPED: OpLifetimeStart %[[#IntVar]] 4
; CHECK: OpStore %[[#IntVar]]
; TYPED: OpLifetimeStop %[[#IntVar]] 0
; UNTYPED: OpLifetimeStop %[[#IntVar]] 4
define spir_func void @int_var() {
  %var = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %var)
  store i32 0, ptr %var, align 4
  call void @llvm.lifetime.end.p0(ptr %var)
  ret void
}

; CHECK-LABEL: Begin function array_var
; TYPED: %[[#ArrVar:]] = OpVariable
; TYPED: OpLifetimeStart %[[#ArrVar]] 0
; UNTYPED: %[[#ArrVar:]] = OpUntypedVariableKHR
; UNTYPED: OpLifetimeStart %[[#ArrVar]] 16
; CHECK: OpStore
; TYPED: OpLifetimeStop %[[#ArrVar]] 0
; UNTYPED: OpLifetimeStop %[[#ArrVar]] 16
define spir_func void @array_var() {
  %var = alloca [4 x float], align 4
  call void @llvm.lifetime.start.p0(ptr %var)
  store [4 x float] zeroinitializer, ptr %var, align 4
  call void @llvm.lifetime.end.p0(ptr %var)
  ret void
}

; CHECK-LABEL: Begin function struct_var
; TYPED: %[[#StructVar:]] = OpVariable
; TYPED: OpLifetimeStart %[[#StructVar]] 0
; UNTYPED: %[[#StructVar:]] = OpUntypedVariableKHR
; UNTYPED: OpLifetimeStart %[[#StructVar]] 16
; CHECK: OpStore %[[#StructVar]]
; TYPED: OpLifetimeStop %[[#StructVar]] 0
; UNTYPED: OpLifetimeStop %[[#StructVar]] 16
define spir_func void @struct_var() {
  %var = alloca %struct, align 8
  call void @llvm.lifetime.start.p0(ptr %var)
  store %struct zeroinitializer, ptr %var, align 8
  call void @llvm.lifetime.end.p0(ptr %var)
  ret void
}

declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)
