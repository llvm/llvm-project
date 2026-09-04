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
; CHECK-DAG: OpConstant %[[#F32]] -0x1.8p+128{{$}}
; CHECK-DAG: OpConstant %[[#F32]] 0x1.4p+128{{$}}
; CHECK-DAG: OpConstant %[[#F64]] 0x1p+1024{{$}}
; CHECK-DAG: OpConstant %[[#F64]] -0x1p+1024{{$}}
; CHECK-DAG: OpConstant %[[#F64]] 0x1.8p+1024{{$}}
; CHECK-DAG: OpConstant %[[#F64]] 0x1.0000000000001p+1024{{$}}

define void @main() {
entry:
  %f = alloca float, align 4
  store float 5.000000e-01, ptr %f, align 4
  %d = alloca double, align 8
  store double 5.000000e-01, ptr %d, align 8
  %hexf = alloca float, align 4
  store float 0x37D5C73200000000, ptr %hexf, align 4
  %inf = alloca float, align 4
  store float +inf, ptr %inf, align 4
  %ninf = alloca float, align 4
  store float -inf, ptr %ninf, align 4
  %nan = alloca float, align 4
  store float 0x7FF8000000000000, ptr %nan, align 4
  %nnan = alloca float, align 4
  store float 0xFFF8000000000000, ptr %nnan, align 4
  %snan = alloca float, align 4
  store float 0x7FF4000000000000, ptr %snan, align 4
  %dinf = alloca double, align 8
  store double 0x7FF0000000000000, ptr %dinf, align 8
  %dninf = alloca double, align 8
  store double 0xFFF0000000000000, ptr %dninf, align 8
  %dnan = alloca double, align 8
  store double 0x7FF8000000000000, ptr %dnan, align 8
  %dsnan = alloca double, align 8
  store double 0x7FF0000000000001, ptr %dsnan, align 8
  ret void
}
