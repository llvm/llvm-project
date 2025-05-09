; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:     %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:    %[[#float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_fp:]] = OpTypePointer Function %[[#float]]
; CHECK-DAG: %[[#float_pp:]] = OpTypePointer Private %[[#float]]
; CHECK-DAG:  %[[#uint_fp:]] = OpTypePointer Function %[[#uint]]
; CHECK-DAG:   %[[#uint_0:]] = OpConstant %[[#uint]] 0
; CHECK-DAG:       %[[#sf:]] = OpTypeStruct %[[#float]]
; CHECK-DAG:       %[[#su:]] = OpTypeStruct %[[#uint]]
; CHECK-DAG:       %[[#sfuf:]] = OpTypeStruct %[[#float]] %[[#uint]] %[[#float]]
; CHECK-DAG:    %[[#sf_fp:]] = OpTypePointer Function %[[#sf]]
; CHECK-DAG:    %[[#su_fp:]] = OpTypePointer Function %[[#su]]
; CHECK-DAG:    %[[#sfuf_fp:]] = OpTypePointer Function %[[#sfuf]]
; CHECK-DAG:    %[[#sfuf_pp:]] = OpTypePointer Private %[[#sfuf]]

%struct.SF = type { float }
%struct.SU = type { i32 }
%struct.SFUF = type { float, i32, float }

@gsfuf = external addrspace(10) global %struct.SFUF
; CHECK: %[[#gsfuf:]] = OpVariable %[[#sfuf_pp]] Private

define internal spir_func float @foo() {
  %1 = alloca %struct.SF, align 4
; CHECK: %[[#var:]]  = OpVariable %[[#sf_fp]] Function

  %2 = load float, ptr %1, align 4
; CHECK: %[[#tmp:]]  = OpAccessChain %[[#float_fp]] %[[#var]] %[[#uint_0]]
; CHECK: %[[#val:]]  = OpLoad %[[#float]] %[[#tmp]] Aligned 4

  ret float %2
}

define internal spir_func i32 @bar() {
  %1 = alloca %struct.SU, align 4
; CHECK: %[[#var:]]  = OpVariable %[[#su_fp]] Function

  %2 = load i32, ptr %1, align 4
; CHECK: %[[#tmp:]]  = OpAccessChain %[[#uint_fp]] %[[#var]] %[[#uint_0]]
; CHECK: %[[#val:]]  = OpLoad %[[#uint]] %[[#tmp]] Aligned 4

  ret i32 %2
}

define internal spir_func float @baz() {
  %1 = alloca %struct.SFUF, align 4
; CHECK: %[[#var:]]  = OpVariable %[[#sfuf_fp]] Function

  %2 = load float, ptr %1, align 4
; CHECK: %[[#tmp:]]  = OpAccessChain %[[#float_fp]] %[[#var]] %[[#uint_0]]
; CHECK: %[[#val:]]  = OpLoad %[[#float]] %[[#tmp]] Aligned 4

  ret float %2
}

define internal spir_func float @biz() {
  %2 = load float, ptr addrspace(10) @gsfuf, align 4
; CHECK: %[[#tmp:]]  = OpAccessChain %[[#float_pp]] %[[#gsfuf]] %[[#uint_0]]
; CHECK: %[[#val:]]  = OpLoad %[[#float]] %[[#tmp]] Aligned 4

  ret float %2
}
