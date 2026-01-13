; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

@i = global i32 0, align 4
@absi = global i32 0, align 4
@f = global float 0.0, align 4
@absf = global float 0.0, align 4

define void @main() #1 {
entry:
  %0 = load i32, ptr @i, align 4

; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] SAbs %[[#]]
  %elt.abs = call i32 @llvm.abs.i32(i32 %0, i1 false)

  store i32 %elt.abs, ptr @absi, align 4
  %1 = load float, ptr @f, align 4

; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] FAbs %[[#]]
  %elt.abs1 = call float @llvm.fabs.f32(float %1)

  store float %elt.abs1, ptr @absf, align 4
  ret void
}

declare i32 @llvm.abs.i32(i32, i1 immarg) #2
declare float @llvm.fabs.f32(float) #2
