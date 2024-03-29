; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; Note: The validator is wrong it wants the return to be a bool vector when it is bool scalar return
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure spirv operation function calls for all are generated.

; CHECK: OpMemoryModel Logical GLSL450

define noundef i1 @all_bool(i1 noundef %a) {
entry:
  ; CHECK: %[[#]] = OpAll %[[#]] %[[#]]
  %hlsl.all = call i1 @llvm.spv.all.i1(i1 %a)
  ret i1 %hlsl.all
}

define noundef i1 @all_int64_t(i64 noundef %p0) {
entry:
  %p0.addr = alloca i64, align 8
  store i64 %p0, ptr %p0.addr, align 8
  %0 = load i64, ptr %p0.addr, align 8
  ; CHECK: %[[#]] = OpAll %[[#]] %[[#]]
  %hlsl.all = call i1 @llvm.spv.all.i64(i64 %0)
  ret i1 %hlsl.all
}


define noundef i1 @all_int(i32 noundef %p0) {
entry:
  %p0.addr = alloca i32, align 4
  store i32 %p0, ptr %p0.addr, align 4
  %0 = load i32, ptr %p0.addr, align 4
  ; CHECK: %[[#]] = OpAll %[[#]] %[[#]]
  %hlsl.all = call i1 @llvm.spv.all.i32(i32 %0)
  ret i1 %hlsl.all
}


define noundef i1 @all_int16_t(i16 noundef %p0) {
entry:
  %p0.addr = alloca i16, align 2
  store i16 %p0, ptr %p0.addr, align 2
  %0 = load i16, ptr %p0.addr, align 2
  ; CHECK: %[[#]] = OpAll %[[#]] %[[#]]
  %hlsl.all = call i1 @llvm.spv.all.i16(i16 %0)
  ret i1 %hlsl.all
}

define noundef i1 @all_double(double noundef %p0) {
entry:
  %p0.addr = alloca double, align 8
  store double %p0, ptr %p0.addr, align 8
  %0 = load double, ptr %p0.addr, align 8
  ; CHECK: %[[#]] = OpAll %[[#]] %[[#]]
  %hlsl.all = call i1 @llvm.spv.all.f64(double %0)
  ret i1 %hlsl.all
}


define noundef i1 @all_float(float noundef %p0) {
entry:
  %p0.addr = alloca float, align 4
  store float %p0, ptr %p0.addr, align 4
  %0 = load float, ptr %p0.addr, align 4
  ; CHECK: %[[#]] = OpAll %[[#]] %[[#]]
  %hlsl.all = call i1 @llvm.spv.all.f32(float %0)
  ret i1 %hlsl.all
}


define noundef i1 @all_half(half noundef %p0) {
entry:
  %p0.addr = alloca half, align 2
  store half %p0, ptr %p0.addr, align 2
  %0 = load half, ptr %p0.addr, align 2
  ; CHECK: %[[#]] = OpAll %[[#]] %[[#]]
  %hlsl.all = call i1 @llvm.spv.all.f16(half %0)
  ret i1 %hlsl.all
}


define noundef i1 @all_bool4(<4 x i1> noundef %p0) {
entry:
  ; CHECK: %[[#]] = OpAll %[[#]] %[[#]]
  %hlsl.all = call i1 @llvm.spv.all.v4i1(<4 x i1> %p0)
  ret i1 %hlsl.all
}

declare i1 @llvm.spv.all.v4i1(<4 x i1>)
declare i1 @llvm.spv.all.i1(i1)
declare i1 @llvm.spv.all.i16(i16)
declare i1 @llvm.spv.all.i32(i32)
declare i1 @llvm.spv.all.i64(i64)
declare i1 @llvm.spv.all.f16(half)
declare i1 @llvm.spv.all.f32(float)
declare i1 @llvm.spv.all.f64(double)
