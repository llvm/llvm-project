;; Check that llvm.bitreverse.* intrinsics are lowered for
;; 2/4-bit scalar and vector types.

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_integers,+SPV_KHR_bit_instructions %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_arbitrary_precision_integers,+SPV_KHR_bit_instructions %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability ArbitraryPrecisionIntegersINTEL
; CHECK: OpExtension "SPV_INTEL_arbitrary_precision_integers"

; CHECK: %[[#I4:]] = OpTypeInt 4 0
; CHECK: %[[#I2:]] = OpTypeInt 2 0
; CHECK: %[[#Z4:]] = OpConstantNull %[[#I4]]
; CHECK: %[[#Z2:]] = OpConstantNull %[[#I2]]
; CHECK: %[[#V2I2:]] = OpTypeVector %[[#I2]] 2
; CHECK: %[[#V2I4:]] = OpTypeVector %[[#I4]] 2
; CHECK: %[[#V3I2:]] = OpTypeVector %[[#I2]] 3
; CHECK: %[[#V3I4:]] = OpTypeVector %[[#I4]] 3
; CHECK: %[[#V4I2:]] = OpTypeVector %[[#I2]] 4
; CHECK: %[[#V4I4:]] = OpTypeVector %[[#I4]] 4
; CHECK: %[[#V8I2:]] = OpTypeVector %[[#I2]] 8
; CHECK: %[[#V8I4:]] = OpTypeVector %[[#I4]] 8
; CHECK: %[[#V16I2:]] = OpTypeVector %[[#I2]] 16
; CHECK: %[[#V16I4:]] = OpTypeVector %[[#I4]] 16


; CHECK: %[[#]] = OpBitReverse %[[#I2]] %[[#Z2]]
; CHECK: %[[#]] = OpBitReverse %[[#I4]] %[[#Z4]]
; CHECK: %[[#]] = OpBitReverse %[[#V2I2]] %[[#]]
; CHECK: %[[#]] = OpBitReverse %[[#V2I4]] %[[#]]
; CHECK: %[[#]] = OpBitReverse %[[#V3I2]] %[[#]]
; CHECK: %[[#]] = OpBitReverse %[[#V3I4]] %[[#]]
; CHECK: %[[#]] = OpBitReverse %[[#V4I2]] %[[#]]
; CHECK: %[[#]] = OpBitReverse %[[#V4I4]] %[[#]]
; CHECK: %[[#]] = OpBitReverse %[[#V8I2]] %[[#]]
; CHECK: %[[#]] = OpBitReverse %[[#V8I4]] %[[#]]
; CHECK: %[[#]] = OpBitReverse %[[#V16I2]] %[[#]]
; CHECK: %[[#]] = OpBitReverse %[[#V16I4]] %[[#]]

define spir_kernel void @testBitRev() {
entry:
  %call2 = call i2 @llvm.bitreverse.i2(i2 0)
  %call4 = call i4 @llvm.bitreverse.i4(i4 0)
  ret void
}

define spir_kernel void @testBitRevV2(<2 x i2> %a, <2 x i4> %b) {
entry:
  %call2 = call <2 x i2> @llvm.bitreverse.v2i2(<2 x i2> %a)
  %call4 = call <2 x i4> @llvm.bitreverse.v2i4(<2 x i4> %b)
  ret void
}

define spir_kernel void @testBitRevV3(<3 x i2> %a, <3 x i4> %b) {
entry:
  %call2 = call <3 x i2> @llvm.bitreverse.v3i2(<3 x i2> %a)
  %call4 = call <3 x i4> @llvm.bitreverse.v3i4(<3 x i4> %b)
  ret void
}

define spir_kernel void @testBitRevV4(<4 x i2> %a, <4 x i4> %b) {
entry:
  %call2 = call <4 x i2> @llvm.bitreverse.v4i2(<4 x i2> %a)
  %call4 = call <4 x i4> @llvm.bitreverse.v4i4(<4 x i4> %b)
  ret void
}

define spir_kernel void @testBitRevV8(<8 x i2> %a, <8 x i4> %b) {
entry:
  %call2 = call <8 x i2> @llvm.bitreverse.v8i2(<8 x i2> %a)
  %call4 = call <8 x i4> @llvm.bitreverse.v8i4(<8 x i4> %b)
  ret void
}

define spir_kernel void @testBitRevV16(<16 x i2> %a, <16 x i4> %b) {
entry:
  %call2 = call <16 x i2> @llvm.bitreverse.v16i2(<16 x i2> %a)
  %call4 = call <16 x i4> @llvm.bitreverse.v16i4(<16 x i4> %b)
  ret void
}

declare i2 @llvm.bitreverse.i2(i2)
declare i4 @llvm.bitreverse.i4(i4)
declare <2 x i2> @llvm.bitreverse.v2i2(<2 x i2>)
declare <2 x i4> @llvm.bitreverse.v2i4(<2 x i4>)
declare <3 x i2> @llvm.bitreverse.v3i2(<3 x i2>)
declare <3 x i4> @llvm.bitreverse.v3i4(<3 x i4>)
declare <4 x i2> @llvm.bitreverse.v4i2(<4 x i2>)
declare <4 x i4> @llvm.bitreverse.v4i4(<4 x i4>)
declare <8 x i2> @llvm.bitreverse.v8i2(<8 x i2>)
declare <8 x i4> @llvm.bitreverse.v8i4(<8 x i4>)
declare <16 x i2> @llvm.bitreverse.v16i2(<16 x i2>)
declare <16 x i4> @llvm.bitreverse.v16i4(<16 x i4>)
