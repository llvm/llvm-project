; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define void @main() #1 {
entry:
  %i = alloca i32, align 4
  %absi = alloca i32, align 4
  %f = alloca float, align 4
  %absf = alloca float, align 4
  %0 = load i32, ptr %i, align 4

; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] SAbs %[[#]]
  %elt.abs = call i32 @llvm.abs.i32(i32 %0, i1 false)

  store i32 %elt.abs, ptr %absi, align 4
  %1 = load float, ptr %f, align 4

; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] FAbs %[[#]]
  %elt.abs1 = call float @llvm.fabs.f32(float %1)

  store float %elt.abs1, ptr %absf, align 4
  ret void
}

declare i32 @llvm.abs.i32(i32, i1 immarg) #2
declare float @llvm.fabs.f32(float) #2
