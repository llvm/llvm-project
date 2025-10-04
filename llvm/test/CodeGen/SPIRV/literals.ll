; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#F64:]] = OpTypeFloat 64
; CHECK-DAG: OpConstant %[[#F32]] 0.5{{$}}
; CHECK-DAG: OpConstant %[[#F64]] 0.5{{$}}
; CHECK-DAG: OpConstant %[[#F32]] 1.0000016166037976e-39{{$}}
; CHECK-DAG: OpConstant %[[#F32]] 0x1p+128{{$}}
; CHECK-DAG: OpConstant %[[#F32]] -0x1p+128{{$}}
; CHECK-DAG: OpConstant %[[#F32]] 0x1.8p+128{{$}}

define void @main() {
entry:
  %f = alloca float, align 4
  store float 5.000000e-01, ptr %f, align 4
  %d = alloca double, align 8
  store double 5.000000e-01, ptr %d, align 8
  %hexf = alloca float, align 4
  store float 0x37D5C73200000000, ptr %hexf, align 4
  %inf = alloca float, align 4
  store float 0x7FF0000000000000, ptr %inf, align 4
  %ninf = alloca float, align 4
  store float 0xFFF0000000000000, ptr %ninf, align 4
  %nan = alloca float, align 4
  store float 0x7FF8000000000000, ptr %nan, align 4
  ret void
}
