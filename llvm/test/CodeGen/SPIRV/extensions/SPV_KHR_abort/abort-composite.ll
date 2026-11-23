; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

; Verify that OpAbortKHR can consume a composite (struct or vector) message,
; built up via OpCompositeInsert, mirroring the example in the SPV_KHR_abort
; specification.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#F32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#STRUCT:]] = OpTypeStruct %[[#I32]] %[[#I32]] %[[#F32]]
; CHECK-DAG: %[[#VEC:]] = OpTypeVector %[[#I32]] 4

; CHECK: %[[#S0:]] = OpCompositeInsert %[[#STRUCT]] %{{[0-9]+}} %{{[0-9]+}} 0
; CHECK: %[[#S1:]] = OpCompositeInsert %[[#STRUCT]] %{{[0-9]+}} %[[#S0]] 1
; CHECK: %[[#S2:]] = OpCompositeInsert %[[#STRUCT]] %{{[0-9]+}} %[[#S1]] 2
; CHECK: OpAbortKHR %[[#STRUCT]] %[[#S2]]

; CHECK: %[[#V0:]] = OpCompositeInsert %[[#VEC]] %{{[0-9]+}} %{{[0-9]+}} 0
; CHECK: %[[#V1:]] = OpCompositeInsert %[[#VEC]] %{{[0-9]+}} %[[#V0]] 1
; CHECK: %[[#V2:]] = OpCompositeInsert %[[#VEC]] %{{[0-9]+}} %[[#V1]] 2
; CHECK: %[[#V3:]] = OpCompositeInsert %[[#VEC]] %{{[0-9]+}} %[[#V2]] 3
; CHECK: OpAbortKHR %[[#VEC]] %[[#V3]]

%struct.Msg = type { i32, i32, float }

declare void @_Z16__spirv_AbortKHR3Msg(%struct.Msg) #0
declare void @_Z16__spirv_AbortKHRDv4_j(<4 x i32>) #0

define spir_kernel void @abort_with_struct(i32 %x, i32 %y, float %z) {
entry:
  %m0 = insertvalue %struct.Msg poison, i32 %x, 0
  %m1 = insertvalue %struct.Msg %m0, i32 %y, 1
  %m2 = insertvalue %struct.Msg %m1, float %z, 2
  call void @_Z16__spirv_AbortKHR3Msg(%struct.Msg %m2)
  unreachable
}

define spir_kernel void @abort_with_vector(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %v0 = insertelement <4 x i32> poison, i32 %a, i32 0
  %v1 = insertelement <4 x i32> %v0, i32 %b, i32 1
  %v2 = insertelement <4 x i32> %v1, i32 %c, i32 2
  %v3 = insertelement <4 x i32> %v2, i32 %d, i32 3
  call void @_Z16__spirv_AbortKHRDv4_j(<4 x i32> %v3)
  unreachable
}

attributes #0 = { noreturn }
