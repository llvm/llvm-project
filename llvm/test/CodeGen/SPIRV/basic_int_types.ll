; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define void @main() {
entry:
; CHECK-DAG:   %[[#short:]] = OpTypeInt 16 0
; CHECK-DAG:     %[[#int:]] = OpTypeInt 32 0
; CHECK-DAG:    %[[#long:]] = OpTypeInt 64 0

; CHECK-DAG: %[[#v2short:]] = OpTypeVector %[[#short]] 2
; CHECK-DAG: %[[#v3short:]] = OpTypeVector %[[#short]] 3
; CHECK-DAG: %[[#v4short:]] = OpTypeVector %[[#short]] 4

; CHECK-DAG:   %[[#v2int:]] = OpTypeVector %[[#int]] 2
; CHECK-DAG:   %[[#v3int:]] = OpTypeVector %[[#int]] 3
; CHECK-DAG:   %[[#v4int:]] = OpTypeVector %[[#int]] 4

; CHECK-DAG:  %[[#v2long:]] = OpTypeVector %[[#long]] 2
; CHECK-DAG:  %[[#v3long:]] = OpTypeVector %[[#long]] 3
; CHECK-DAG:  %[[#v4long:]] = OpTypeVector %[[#long]] 4

; CHECK-DAG:   %[[#ptr_Function_short:]] = OpTypePointer Function %[[#short]]
; CHECK-DAG:     %[[#ptr_Function_int:]] = OpTypePointer Function %[[#int]]
; CHECK-DAG:    %[[#ptr_Function_long:]] = OpTypePointer Function %[[#long]]
; CHECK-DAG: %[[#ptr_Function_v2short:]] = OpTypePointer Function %[[#v2short]]
; CHECK-DAG: %[[#ptr_Function_v3short:]] = OpTypePointer Function %[[#v3short]]
; CHECK-DAG: %[[#ptr_Function_v4short:]] = OpTypePointer Function %[[#v4short]]
; CHECK-DAG:   %[[#ptr_Function_v2int:]] = OpTypePointer Function %[[#v2int]]
; CHECK-DAG:   %[[#ptr_Function_v3int:]] = OpTypePointer Function %[[#v3int]]
; CHECK-DAG:   %[[#ptr_Function_v4int:]] = OpTypePointer Function %[[#v4int]]
; CHECK-DAG:  %[[#ptr_Function_v2long:]] = OpTypePointer Function %[[#v2long]]
; CHECK-DAG:  %[[#ptr_Function_v3long:]] = OpTypePointer Function %[[#v3long]]
; CHECK-DAG:  %[[#ptr_Function_v4long:]] = OpTypePointer Function %[[#v4long]]

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_short]] Function
  %int16_t_Val = alloca i16, align 2

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_int]] Function
  %int_Val = alloca i32, align 4

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_long]] Function
  %int64_t_Val = alloca i64, align 8

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v2short]] Function
  %int16_t2_Val = alloca <2 x i16>, align 4

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v3short]] Function
  %int16_t3_Val = alloca <3 x i16>, align 8

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v4short]] Function
  %int16_t4_Val = alloca <4 x i16>, align 8

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v2int]] Function
  %int2_Val = alloca <2 x i32>, align 8

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v3int]] Function
  %int3_Val = alloca <3 x i32>, align 16

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v4int]] Function
  %int4_Val = alloca <4 x i32>, align 16

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v2long]] Function
  %int64_t2_Val = alloca <2 x i64>, align 16

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v3long]] Function
  %int64_t3_Val = alloca <3 x i64>, align 32

; CHECK: %[[#]] = OpVariable %[[#ptr_Function_v4long]] Function
  %int64_t4_Val = alloca <4 x i64>, align 32

  ret void
}
