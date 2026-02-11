; RUN: llc -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_fma %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=spirv64-unknown-unknown < %s | FileCheck --check-prefix=CHECK-NO-EXT %s
; TODO: Add spirv-val validation once the extension is supported.

; CHECK: OpCapability FmaKHR
; CHECK: OpExtension "SPV_KHR_fma"
; CHECK: %[[#TYPE_FLOAT:]] = OpTypeFloat 32
; CHECK: %[[#TYPE_VEC:]] = OpTypeVector %[[#TYPE_FLOAT]] 4
; CHECK: OpFmaKHR %[[#TYPE_FLOAT]] %[[#]]
; CHECK: OpFmaKHR %[[#TYPE_VEC]] %[[#]]
; CHECK: OpFmaKHR %[[#TYPE_FLOAT]] %[[#]]

; CHECK-NO-EXT-NOT: OpCapability FmaKHR
; CHECK-NO-EXT-NOT: OpExtension "SPV_KHR_fma"
; CHECK-NO-EXT: %[[#TYPE_FLOAT:]] = OpTypeFloat 32
; CHECK-NO-EXT: %[[#TYPE_VEC:]] = OpTypeVector %[[#TYPE_FLOAT]] 4
; CHECK-NO-EXT: OpExtInst %[[#TYPE_FLOAT]] %[[#]] fma
; CHECK-NO-EXT: OpExtInst %[[#TYPE_VEC]] %[[#]] fma
; CHECK-NO-EXT: OpExtInst %[[#TYPE_FLOAT]] %[[#]] fma

define spir_func float @test_fma_scalar(float %a, float %b, float %c) {
entry:
  %result = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %result
}

define spir_func <4 x float> @test_fma_vector(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
entry:
  %result = call <4 x float> @llvm.fma.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %result
}

; Case to test fma translation via OCL builtins.
define spir_func float @test_fma_ocl_scalar(float %a, float %b, float %c) {
entry:
  %result = call spir_func float @_Z15__spirv_ocl_fmafff(float %a, float %b, float %c)
  ret float %result
}

declare float @llvm.fma.f32(float, float, float)
declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare spir_func float @_Z15__spirv_ocl_fmafff(float, float, float)
