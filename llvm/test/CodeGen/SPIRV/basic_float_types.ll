; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define void @main() {
entry:

; CHECK-DAG: OpCapability Float16
; CHECK-DAG: OpCapability Float64

; CHECK-DAG:     %[[#half:]] = OpTypeFloat 16
; CHECK-DAG:    %[[#float:]] = OpTypeFloat 32
; CHECK-DAG:   %[[#double:]] = OpTypeFloat 64

; CHECK-DAG:   %[[#v2half:]] = OpTypeVector %[[#half]] 2
; CHECK-DAG:   %[[#v3half:]] = OpTypeVector %[[#half]] 3
; CHECK-DAG:   %[[#v4half:]] = OpTypeVector %[[#half]] 4

; CHECK-DAG:  %[[#v2float:]] = OpTypeVector %[[#float]] 2
; CHECK-DAG:  %[[#v3float:]] = OpTypeVector %[[#float]] 3
; CHECK-DAG:  %[[#v4float:]] = OpTypeVector %[[#float]] 4

; CHECK-DAG: %[[#v2double:]] = OpTypeVector %[[#double]] 2
; CHECK-DAG: %[[#v3double:]] = OpTypeVector %[[#double]] 3
; CHECK-DAG: %[[#v4double:]] = OpTypeVector %[[#double]] 4

; CHECK-DAG:     %[[#ptr_Function_half:]] = OpTypePointer Function %[[#half]]
; CHECK-DAG:    %[[#ptr_Function_float:]] = OpTypePointer Function %[[#float]]
; CHECK-DAG:   %[[#ptr_Function_double:]] = OpTypePointer Function %[[#double]]
; CHECK-DAG:   %[[#ptr_Function_v2half:]] = OpTypePointer Function %[[#v2half]]
; CHECK-DAG:   %[[#ptr_Function_v3half:]] = OpTypePointer Function %[[#v3half]]
; CHECK-DAG:   %[[#ptr_Function_v4half:]] = OpTypePointer Function %[[#v4half]]
; CHECK-DAG:  %[[#ptr_Function_v2float:]] = OpTypePointer Function %[[#v2float]]
; CHECK-DAG:  %[[#ptr_Function_v3float:]] = OpTypePointer Function %[[#v3float]]
; CHECK-DAG:  %[[#ptr_Function_v4float:]] = OpTypePointer Function %[[#v4float]]
; CHECK-DAG: %[[#ptr_Function_v2double:]] = OpTypePointer Function %[[#v2double]]
; CHECK-DAG: %[[#ptr_Function_v3double:]] = OpTypePointer Function %[[#v3double]]
; CHECK-DAG: %[[#ptr_Function_v4double:]] = OpTypePointer Function %[[#v4double]]

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_half]] Function
  %half_Val = alloca half, align 2

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_float]] Function
  %float_Val = alloca float, align 4

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_double]] Function
  %double_Val = alloca double, align 8

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v2half]] Function
  %half2_Val = alloca <2 x half>, align 4

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v3half]] Function
  %half3_Val = alloca <3 x half>, align 8

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v4half]] Function
  %half4_Val = alloca <4 x half>, align 8

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v2float]] Function
  %float2_Val = alloca <2 x float>, align 8

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v3float]] Function
  %float3_Val = alloca <3 x float>, align 16

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v4float]] Function
  %float4_Val = alloca <4 x float>, align 16

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v2double]] Function
  %double2_Val = alloca <2 x double>, align 16

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v3double]] Function
  %double3_Val = alloca <3 x double>, align 32

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v4double]] Function
  %double4_Val = alloca <4 x double>, align 32
  ret void
}
