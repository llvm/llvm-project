; OpLifetimeStart/OpLifetimeStop must have Size 0 when Pointer points to a
; non-void type. For typed allocas the backend should use Size 0 directly on the
; OpVariable rather than bitcasting it to uchar* to carry a byte size. In
; particular, lifetime markers alone must not introduce a uchar type or the
; Int8 capability.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=OpBitcast
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --implicit-check-not=OpBitcast
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: OpCapability Int8
; CHECK-NOT: OpTypeInt 8

%struct = type { i64, float }

; CHECK-LABEL: Begin function int_var
; CHECK: %[[#IntVar:]] = OpVariable
; CHECK: OpLifetimeStart %[[#IntVar]] 0
; CHECK: OpStore %[[#IntVar]]
; CHECK: OpLifetimeStop %[[#IntVar]] 0
define spir_func void @int_var() {
  %var = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %var)
  store i32 0, ptr %var, align 4
  call void @llvm.lifetime.end.p0(ptr %var)
  ret void
}

; CHECK-LABEL: Begin function array_var
; CHECK: %[[#ArrVar:]] = OpVariable
; CHECK: OpLifetimeStart %[[#ArrVar]] 0
; CHECK: OpStore
; CHECK: OpLifetimeStop %[[#ArrVar]] 0
define spir_func void @array_var() {
  %var = alloca [4 x float], align 4
  call void @llvm.lifetime.start.p0(ptr %var)
  store [4 x float] zeroinitializer, ptr %var, align 4
  call void @llvm.lifetime.end.p0(ptr %var)
  ret void
}

; CHECK-LABEL: Begin function struct_var
; CHECK: %[[#StructVar:]] = OpVariable
; CHECK: OpLifetimeStart %[[#StructVar]] 0
; CHECK: OpStore %[[#StructVar]]
; CHECK: OpLifetimeStop %[[#StructVar]] 0
define spir_func void @struct_var() {
  %var = alloca %struct, align 8
  call void @llvm.lifetime.start.p0(ptr %var)
  store %struct zeroinitializer, ptr %var, align 8
  call void @llvm.lifetime.end.p0(ptr %var)
  ret void
}

declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.lifetime.end.p0(ptr nocapture)
