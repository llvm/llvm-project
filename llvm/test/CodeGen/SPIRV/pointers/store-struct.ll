; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:      %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:     %[[#float:]] = OpTypeFloat 32
; CHECK-DAG:  %[[#float_fp:]] = OpTypePointer Function %[[#float]]
; CHECK-DAG:  %[[#float_pp:]] = OpTypePointer Private %[[#float]]
; CHECK-DAG:   %[[#uint_fp:]] = OpTypePointer Function %[[#uint]]
; CHECK-DAG:    %[[#uint_0:]] = OpConstant %[[#uint]] 0
; CHECK-DAG:    %[[#uint_4:]] = OpConstant %[[#uint]] 4
; CHECK-DAG:    %[[#float_0:]] = OpConstant %[[#float]] 0
; CHECK-DAG:        %[[#sf:]] = OpTypeStruct %[[#float]]
; CHECK-DAG:        %[[#su:]] = OpTypeStruct %[[#uint]]
; CHECK-DAG:       %[[#ssu:]] = OpTypeStruct %[[#su]]
; CHECK-DAG:        %[[#sfuf:]] = OpTypeStruct %[[#float]] %[[#uint]] %[[#float]]
; CHECK-DAG:        %[[#uint4:]] = OpTypeVector %[[#uint]] 4
; CHECK-DAG:        %[[#sv:]] = OpTypeStruct %[[#uint4]]
; CHECK-DAG:        %[[#ssv:]] = OpTypeStruct %[[#sv]]
; CHECK-DAG:        %[[#assv:]] = OpTypeArray %[[#ssv]] %[[#uint_4]]
; CHECK-DAG:        %[[#sassv:]] = OpTypeStruct %[[#assv]]
; CHECK-DAG:        %[[#ssassv:]] = OpTypeStruct %[[#sassv]]
; CHECK-DAG:     %[[#sf_fp:]] = OpTypePointer Function %[[#sf]]
; CHECK-DAG:     %[[#su_fp:]] = OpTypePointer Function %[[#su]]
; CHECK-DAG:    %[[#ssu_fp:]] = OpTypePointer Function %[[#ssu]]
; CHECK-DAG:    %[[#ssv_fp:]] = OpTypePointer Function %[[#ssv]]
; CHECK-DAG: %[[#ssassv_fp:]] = OpTypePointer Function %[[#ssassv]]
; CHECK-DAG:   %[[#sfuf_fp:]] = OpTypePointer Function %[[#sfuf]]
; CHECK-DAG:   %[[#sfuf_pp:]] = OpTypePointer Private %[[#sfuf]]

%struct.SF = type { float }
%struct.SU = type { i32 }
%struct.SFUF = type { float, i32, float }
%struct.SSU = type { %struct.SU }
%struct.SV = type { <4 x i32> }
%struct.SSV = type { %struct.SV }
%struct.SASSV = type { [4 x %struct.SSV] }
%struct.SSASSV = type { %struct.SASSV }

@gsfuf = external addrspace(10) global %struct.SFUF
; CHECK-DAG: %[[#gsfuf:]] = OpVariable %[[#sfuf_pp]] Private

define internal spir_func void @foo() {
  %1 = alloca %struct.SF, align 4
; CHECK: %[[#var:]]  = OpVariable %[[#sf_fp]] Function

  store float 0.0, ptr %1, align 4
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#float_fp]] %[[#var]] %[[#uint_0]]
; CHECK:               OpStore %[[#tmp]] %[[#float_0]] Aligned 4

  ret void
}

define internal spir_func void @bar() {
  %1 = alloca %struct.SU, align 4
; CHECK: %[[#var:]]  = OpVariable %[[#su_fp]] Function

  store i32 0, ptr %1, align 4
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#uint_fp]] %[[#var]] %[[#uint_0]]
; CHECK:               OpStore %[[#tmp]] %[[#uint_0]] Aligned 4

  ret void
}

define internal spir_func void @baz() {
  %1 = alloca %struct.SFUF, align 4
; CHECK: %[[#var:]]  = OpVariable %[[#sfuf_fp]] Function

  store float 0.0, ptr %1, align 4
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#float_fp]] %[[#var]] %[[#uint_0]]
; CHECK:               OpStore %[[#tmp]] %[[#float_0]] Aligned 4

  ret void
}

define internal spir_func void @biz() {
  store float 0.0, ptr addrspace(10) @gsfuf, align 4
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#float_pp]] %[[#gsfuf]] %[[#uint_0]]
; CHECK:               OpStore %[[#tmp]] %[[#float_0]] Aligned 4

  ret void
}

define internal spir_func void @nested_store() {
  %1 = alloca %struct.SSU, align 4
; CHECK: %[[#var:]]  = OpVariable %[[#ssu_fp]] Function

  store i32 0, ptr %1, align 4
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#uint_fp]] %[[#var]] %[[#uint_0]] %[[#uint_0]]
; CHECK:               OpStore %[[#tmp]] %[[#uint_0]] Aligned 4

  ret void
}

define internal spir_func void @nested_store_vector() {
  %1 = alloca %struct.SSV, align 4
; CHECK: %[[#var:]]  = OpVariable %[[#ssv_fp]] Function

  store i32 0, ptr %1, align 4
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#uint_fp]] %[[#var]] %[[#uint_0]] %[[#uint_0]] %[[#uint_0]]
; CHECK:               OpStore %[[#tmp]] %[[#uint_0]] Aligned 4

  ret void
}

define internal spir_func void @nested_array_vector() {
  %1 = alloca %struct.SSASSV, align 4
; CHECK: %[[#var:]]  = OpVariable %[[#ssassv_fp]] Function

  store i32 0, ptr %1, align 4
; CHECK: %[[#tmp:]]  = OpInBoundsAccessChain %[[#uint_fp]] %[[#var]] %[[#uint_0]] %[[#uint_0]] %[[#uint_0]] %[[#uint_0]] %[[#uint_0]] %[[#uint_0]]
; CHECK:               OpStore %[[#tmp]] %[[#uint_0]] Aligned 4

  ret void
}
