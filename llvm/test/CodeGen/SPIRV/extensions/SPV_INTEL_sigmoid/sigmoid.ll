; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_sigmoid %s -o - | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; TODO: Enable spirv-val once the extension is published and SPIRV-Headers/SPIRV-Tools can recognize it.
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_sigmoid %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: OpFSigmoidINTEL instruction requires the following SPIR-V extension: SPV_INTEL_sigmoid

; CHECK-DAG: OpCapability SigmoidINTEL
; CHECK-DAG: OpExtension "SPV_INTEL_sigmoid"
; CHECK-DAG: %[[#FP16Ty:]] = OpTypeFloat 16
; CHECK-DAG: %[[#FP16v8Ty:]] = OpTypeVector %[[#FP16Ty]] 8
; CHECK-DAG: %[[#FP32Ty:]] = OpTypeFloat 32
; CHECK-DAG: %[[#FP32v8Ty:]] = OpTypeVector %[[#FP32Ty]] 8
; CHECK-DAG: %[[#FP16Const:]] = OpConstant %[[#FP16Ty]] 15360
; CHECK-DAG: %[[#FP32Const:]] = OpConstant %[[#FP32Ty]] 1

; CHECK: %[[#FP16Val:]] = OpFunctionParameter %[[#FP16Ty]]
; CHECK: %[[#FP16v8Val:]] = OpFunctionParameter %[[#FP16v8Ty]]
; CHECK: %[[#]] = OpFSigmoidINTEL %[[#FP16Ty]] %[[#FP16Val]]
; CHECK: %[[#]] = OpFSigmoidINTEL %[[#FP16v8Ty]] %[[#FP16v8Val]]
; CHECK: %[[#]] = OpFSigmoidINTEL %[[#FP16Ty]] %[[#FP16Const]]

; CHECK: %[[#FP32Val:]] = OpFunctionParameter %[[#FP32Ty]]
; CHECK: %[[#FP32v8Val:]] = OpFunctionParameter %[[#FP32v8Ty]]
; CHECK: %[[#]] = OpFSigmoidINTEL %[[#FP32Ty]] %[[#FP32Val]]
; CHECK: %[[#]] = OpFSigmoidINTEL %[[#FP32v8Ty]] %[[#FP32v8Val]]
; CHECK: %[[#]] = OpFSigmoidINTEL %[[#FP32Ty]] %[[#FP32Const]]

define spir_func void @test_half(half %a, <8 x half> %in) {
  %1 = tail call spir_func half @_Z21__spirv_FSigmoidINTELDh(half %a)
  %2 = tail call spir_func <8 x half> @_Z21__spirv_FSigmoidINTELDv8_Dh(<8 x half> %in)
  %3 = tail call spir_func half @_Z21__spirv_FSigmoidINTELDh(half 1.000000e+00)
  ret void
}

define spir_func void @test_float(float %a, <8 x float> %in) {
  %1 = tail call spir_func float @_Z21__spirv_FSigmoidINTELf(float %a)
  %2 = tail call spir_func <8 x float> @_Z21__spirv_FSigmoidINTELDv8_f(<8 x float> %in)
  %3 = tail call spir_func float @_Z21__spirv_FSigmoidINTELf(float 1.000000e+00)
  ret void
}

declare spir_func half @_Z21__spirv_FSigmoidINTELDh(half)
declare spir_func <8 x half> @_Z21__spirv_FSigmoidINTELDv8_Dh(<8 x half>)
declare spir_func float @_Z21__spirv_FSigmoidINTELf(float)
declare spir_func <8 x float> @_Z21__spirv_FSigmoidINTELDv8_f(<8 x float>)
