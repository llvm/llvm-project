; Check that untyped pointers extension does not affect the translation of images.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_untyped_pointers %s -o - -filetype=obj | spirv-val %}
; XFAIL: *

; CHECK-DAG: [[#I32:]]      = OpTypeInt 32 0
; CHECK-DAG: [[#IMAGETY:]] = OpTypeImage [[#I32]] 2 0 0 0 0 0 0
; CHECK-DAG: [[#IVEC:]]    = OpTypeVector [[#I32]] [[#]]
; CHECK-DAG: [[#F32:]]     = OpTypeFloat 32
; CHECK-DAG: [[#FVEC:]]    = OpTypeVector [[#F32]] [[#]]

; CHECK: [[#IMAGE0:]]  = OpLoad [[#IMAGETY]] [[#]]
; CHECK: [[#COORD0:]]  = OpLoad [[#IVEC]] [[#]]
; CHECK: [[#]]         = OpImageRead [[#IVEC]] [[#IMAGE0]] [[#COORD0]] 8192

; CHECK: [[#IMAGE1:]]  = OpLoad [[#IMAGETY]] [[#]]
; CHECK: [[#COORD1:]]  = OpLoad [[#IVEC]] [[#]]
; CHECK: [[#]]         = OpImageRead [[#FVEC]] [[#IMAGE1]] [[#COORD1]]

define dso_local spir_kernel void @kernelA(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %input) {
entry:
  %input.addr = alloca target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), align 8
  %c = alloca <4 x i32>, align 16
  %.compoundliteral = alloca <4 x i32>, align 16
  store target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %input, ptr %input.addr, align 8
  %0 = load target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), ptr %input.addr, align 8
  store <4 x i32> zeroinitializer, ptr %.compoundliteral, align 16
  %1 = load <4 x i32>, ptr %.compoundliteral, align 16
  %call = call spir_func <4 x i32> @_Z12read_imageui14ocl_image3d_roDv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %0, <4 x i32> noundef %1)
  store <4 x i32> %call, ptr %c, align 16
  ret void
}

declare spir_func <4 x i32> @_Z12read_imageui14ocl_image3d_roDv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), <4 x i32> noundef)

define dso_local spir_kernel void @kernelB(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %input) {
entry:
  %input.addr = alloca target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), align 8
  %f = alloca <4 x float>, align 16
  %.compoundliteral = alloca <4 x i32>, align 16
  store target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %input, ptr %input.addr, align 8
  %0 = load target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), ptr %input.addr, align 8
  store <4 x i32> zeroinitializer, ptr %.compoundliteral, align 16
  %1 = load <4 x i32>, ptr %.compoundliteral, align 16
  %call = call spir_func <4 x float> @_Z11read_imagef14ocl_image3d_roDv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0) %0, <4 x i32> noundef %1)
  store <4 x float> %call, ptr %f, align 16
  ret void
}

declare spir_func <4 x float> @_Z11read_imagef14ocl_image3d_roDv4_i(target("spirv.Image", void, 2, 0, 0, 0, 0, 0, 0), <4 x i32> noundef)
