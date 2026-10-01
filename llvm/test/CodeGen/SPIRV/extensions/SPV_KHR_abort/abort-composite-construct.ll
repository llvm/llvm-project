; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_abort %s -o - -filetype=obj | spirv-val %}

;; Verify that OpAbortKHR can consume a composite produced by
;; OpCompositeConstruct, mirroring the SPV_KHR_abort spec example which builds
;; the message via OpCompositeConstruct.

; CHECK-DAG: OpCapability AbortKHR
; CHECK-DAG: OpExtension "SPV_KHR_abort"
; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#V2:]] = OpTypeVector %[[#I32]] 2

; CHECK:     %[[#CC:]] = OpCompositeConstruct %[[#V2]] %{{[0-9]+}} %{{[0-9]+}}
; CHECK:     OpAbortKHR %[[#V2]] %[[#CC]]
; CHECK-NOT: OpUnreachable

declare void @llvm.spv.abort(<2 x i32>) #0

define spir_kernel void @abort_composite_construct(i32 %a, i32 %b) {
entry:
  %va = insertelement <1 x i32> poison, i32 %a, i32 0
  %vb = insertelement <1 x i32> poison, i32 %b, i32 0
  %v = shufflevector <1 x i32> %va, <1 x i32> %vb,
                     <2 x i32> <i32 0, i32 1>
  call void @llvm.spv.abort(<2 x i32> %v)
  unreachable
}

attributes #0 = { noreturn }
