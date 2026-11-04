;; Check that llvm.bitreverse.* intrinsics are lowered correctly for
;; 2/4-bit scalar and vector types under different extension combinations.
;;
;; Without SPV_ALTERA_arbitrary_precision_integers, sub-byte types are
;; widened to i8 and OpBitReverse operates on the widened type.
;; With SPV_ALTERA, native sub-byte types are preserved:
;;   - Without SPV_KHR_bit_instructions: bitreverse is emulated.
;;   - With SPV_KHR_bit_instructions: native OpBitReverse is used.

;; Widen-to-i8 case (no ALTERA, sub-byte types widened):
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bit_instructions %s -o - | FileCheck %s --check-prefix=CHECK-WIDEN
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_bit_instructions %s -o - | FileCheck %s --check-prefix=CHECK-WIDEN
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_bit_instructions %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_bit_instructions %s -o - -filetype=obj | spirv-val %}

;; Native sub-byte types with emulated bitreverse (ALTERA only):
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s --check-prefix=CHECK-EMULATION
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}

;; Native sub-byte types with native OpBitReverse (ALTERA + KHR):
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers,+SPV_KHR_bit_instructions %s -o - | FileCheck %s --check-prefix=CHECK-NATIVE
; RUN: %if spirv-tools %{ llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers,+SPV_KHR_bit_instructions %s -o - -filetype=obj | spirv-val %}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Capability and type declarations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Widen case: sub-byte types become i8
; CHECK-WIDEN: OpCapability BitInstructions
; CHECK-WIDEN: OpExtension "SPV_KHR_bit_instructions"
; CHECK-WIDEN-DAG: %[[#I8:]] = OpTypeInt 8 0

; CHECK-EMULATION-DAG: OpCapability ArbitraryPrecisionIntegersALTERA
; CHECK-EMULATION-DAG: OpExtension "SPV_ALTERA_arbitrary_precision_integers"

; CHECK-NATIVE-DAG: OpCapability ArbitraryPrecisionIntegersALTERA
; CHECK-NATIVE-DAG: OpExtension "SPV_ALTERA_arbitrary_precision_integers"


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Scalar types - 2, 4 bits
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; CHECK-EMULATION-NOT: OpExtension "SPV_KHR_bit_instructions"
; CHECK-NATIVE-DAG: OpExtension "SPV_KHR_bit_instructions"


; CHECK-NATIVE-DAG: %[[#I2:]] = OpTypeInt 2 0
; CHECK-NATIVE-DAG: %[[#I4:]] = OpTypeInt 4 0

; CHECK-NATIVE-DAG: %[[#V2I2:]] = OpTypeVector %[[#I2]] 2
; CHECK-NATIVE-DAG: %[[#V2I4:]] = OpTypeVector %[[#I4]] 2

; CHECK-NATIVE-DAG: %[[#V4I2:]] = OpTypeVector %[[#I2]] 4
; CHECK-NATIVE-DAG: %[[#V4I4:]] = OpTypeVector %[[#I4]] 4

; CHECK-NATIVE-DAG: %[[#V8I2:]] = OpTypeVector %[[#I2]] 8
; CHECK-NATIVE-DAG: %[[#V8I4:]] = OpTypeVector %[[#I4]] 8

; CHECK-NATIVE-DAG: %[[#V16I2:]] = OpTypeVector %[[#I2]] 16
; CHECK-NATIVE-DAG: %[[#V16I4:]] = OpTypeVector %[[#I4]] 16


; CHECK-EMULATION-DAG: %[[#I2:]] = OpTypeInt 2 0
; CHECK-EMULATION-DAG: %[[#I4:]] = OpTypeInt 4 0

; CHECK-EMULATION-DAG: %[[#V2I2:]] = OpTypeVector %[[#I2]] 2
; CHECK-EMULATION-DAG: %[[#V2I4:]] = OpTypeVector %[[#I4]] 2

; CHECK-EMULATION-DAG: %[[#V4I2:]] = OpTypeVector %[[#I2]] 4
; CHECK-EMULATION-DAG: %[[#V4I4:]] = OpTypeVector %[[#I4]] 4

; CHECK-EMULATION-DAG: %[[#V8I2:]] = OpTypeVector %[[#I2]] 8
; CHECK-EMULATION-DAG: %[[#V8I4:]] = OpTypeVector %[[#I4]] 8

; CHECK-EMULATION-DAG: %[[#V16I2:]] = OpTypeVector %[[#I2]] 16
; CHECK-EMULATION-DAG: %[[#V16I4:]] = OpTypeVector %[[#I4]] 16

define spir_kernel void @test_scalar(i8 %a8, i2 %a2, i4 %a4, ptr addrspace(1) %res) #0 {
entry:
  ; CHECK-WIDEN-LABEL: Begin function test_scalar
  ; CHECK-NATIVE-LABEL: Begin function test_scalar
  ; CHECK-EMULATION-LABEL: Begin function test_scalar
  ; CHECK-EMULATION-NOT: OpBitReverse

  ; CHECK-NATIVE-DAG: %[[#I2_Arg:]] = OpFunctionParameter %[[#I2]]
  ; CHECK-NATIVE-DAG: %[[#I4_Arg:]] = OpFunctionParameter %[[#I4]]

  ; CHECK-EMULATION-DAG: %[[#I2_Arg:]] = OpFunctionParameter %[[#I2]]
  ; CHECK-EMULATION-DAG: %[[#I4_Arg:]] = OpFunctionParameter %[[#I4]]

  ; CHECK-WIDEN: OpBitReverse %[[#I8]] %[[#]]
  ; CHECK-NATIVE: OpBitReverse  %[[#I2]] %[[#I2_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#I2]] %[[#I2_Arg]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#I2:]] %[[#]] %[[#]]

  %call2 = call i2 @llvm.bitreverse.i2(i2 %a2)
  store i2 %call2, ptr addrspace(1) %res, align 2

  ; CHECK-WIDEN: OpBitReverse %[[#I8]] %[[#]]
  ; CHECK-NATIVE: OpBitReverse  %[[#I4]] %[[#I4_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#I4]] %[[#I4_Arg]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#I4:]] %[[#]] %[[#]]

  %call4 = call i4 @llvm.bitreverse.i4(i4 %a4)
  store i4 %call4, ptr addrspace(1) %res, align 4

  ; CHECK-WIDEN-LABEL: OpFunctionEnd
  ; CHECK-NATIVE-LABEL: OpFunctionEnd
  ; CHECK-EMULATION-LABEL: OpFunctionEnd
  ret void
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 2-element vectors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define spir_kernel void @test_vector_2(<2 x i2> %a2, <2 x i4> %a4, ptr addrspace(1) %res) #0 {
entry:
  ; CHECK-NATIVE-LABEL: Begin function test_vector_2
  ; CHECK-EMULATION-LABEL: Begin function test_vector_2
  ; CHECK-EMULATION-NOT: OpBitReverse

  ; CHECK-NATIVE-DAG: %[[#V2I2_Arg:]] = OpFunctionParameter %[[#V2I2]]
  ; CHECK-NATIVE-DAG: %[[#V2I4_Arg:]] = OpFunctionParameter %[[#V2I4]]

  ; CHECK-EMULATION-DAG: %[[#V2I2_Arg:]] = OpFunctionParameter %[[#V2I2]]
  ; CHECK-EMULATION-DAG: %[[#V2I4_Arg:]] = OpFunctionParameter %[[#V2I4]]

  ; CHECK-NATIVE: OpBitReverse  %[[#V2I2]] %[[#V2I2_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V2I2]] %[[#V2I2_Arg]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V2I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V2I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V2I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V2I2:]] %[[#]] %[[#]]
  %call2 = call <2 x i2> @llvm.bitreverse.v2i2(<2 x i2> %a2)
  store <2 x i2> %call2, ptr addrspace(1) %res, align 4

  ; CHECK-NATIVE: OpBitReverse  %[[#V2I4]] %[[#V2I4_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V2I4]] %[[#V2I4_Arg]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V2I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V2I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V2I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V2I4:]] %[[#]] %[[#]]
  %call4 = call <2 x i4> @llvm.bitreverse.v2i4(<2 x i4> %a4)
  store <2 x i4> %call4, ptr addrspace(1) %res, align 8

  ; CHECK-NATIVE-LABEL: OpFunctionEnd
  ; CHECK-EMULATION-LABEL: OpFunctionEnd
  ret void
}


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 4-element vectors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define spir_kernel void @test_vector_4(<4 x i2> %a2, <4 x i4> %a4, ptr addrspace(1) %res) #0 {
entry:
  ; CHECK-NATIVE-LABEL: Begin function test_vector_4
  ; CHECK-EMULATION-LABEL: Begin function test_vector_4
  ; CHECK-EMULATION-NOT: OpBitReverse

  ; CHECK-NATIVE-DAG: %[[#V4I2_Arg:]] = OpFunctionParameter %[[#V4I2]]
  ; CHECK-NATIVE-DAG: %[[#V4I4_Arg:]] = OpFunctionParameter %[[#V4I4]]

  ; CHECK-EMULATION-DAG: %[[#V4I2_Arg:]] = OpFunctionParameter %[[#V4I2]]
  ; CHECK-EMULATION-DAG: %[[#V4I4_Arg:]] = OpFunctionParameter %[[#V4I4]]

  ; CHECK-NATIVE: OpBitReverse  %[[#V4I2]] %[[#V4I2_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V4I2]] %[[#V4I2_Arg]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V4I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V4I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V4I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V4I2:]] %[[#]] %[[#]]
  %call2 = call <4 x i2> @llvm.bitreverse.v4i2(<4 x i2> %a2)
  store <4 x i2> %call2, ptr addrspace(1) %res, align 4

  ; CHECK-NATIVE: OpBitReverse  %[[#V4I4]] %[[#V4I4_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V4I4]] %[[#V4I4_Arg]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V4I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V4I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V4I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V4I4:]] %[[#]] %[[#]]
  %call4 = call <4 x i4> @llvm.bitreverse.v4i4(<4 x i4> %a4)
  store <4 x i4> %call4, ptr addrspace(1) %res, align 8

  ; CHECK-NATIVE-LABEL: OpFunctionEnd
  ; CHECK-EMULATION-LABEL: OpFunctionEnd
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 8-element vectors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define spir_kernel void @test_vector_8(<8 x i2> %a2, <8 x i4> %a4, ptr addrspace(1) %res) #0 {
entry:
  ; CHECK-NATIVE-LABEL: Begin function test_vector_8
  ; CHECK-EMULATION-LABEL: Begin function test_vector_8
  ; CHECK-EMULATION-NOT: OpBitReverse

  ; CHECK-NATIVE-DAG: %[[#V8I2_Arg:]] = OpFunctionParameter %[[#V8I2]]
  ; CHECK-NATIVE-DAG: %[[#V8I4_Arg:]] = OpFunctionParameter %[[#V8I4]]

  ; CHECK-EMULATION-DAG: %[[#V8I2_Arg:]] = OpFunctionParameter %[[#V8I2]]
  ; CHECK-EMULATION-DAG: %[[#V8I4_Arg:]] = OpFunctionParameter %[[#V8I4]]

  ; CHECK-NATIVE: OpBitReverse  %[[#V8I2]] %[[#V8I2_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V8I2]] %[[#V8I2_Arg]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V8I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V8I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V8I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V8I2:]] %[[#]] %[[#]]
  %call2 = call <8 x i2> @llvm.bitreverse.v8i2(<8 x i2> %a2)
  store <8 x i2> %call2, ptr addrspace(1) %res, align 4

  ; CHECK-NATIVE: OpBitReverse  %[[#V8I4]] %[[#V8I4_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V8I4]] %[[#V8I4_Arg]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V8I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V8I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V8I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V8I4:]] %[[#]] %[[#]]
  %call4 = call <8 x i4> @llvm.bitreverse.v8i4(<8 x i4> %a4)
  store <8 x i4> %call4, ptr addrspace(1) %res, align 8

  ; CHECK-NATIVE-LABEL: OpFunctionEnd
  ; CHECK-EMULATION-LABEL: OpFunctionEnd
  ret void
}

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 16-element vectors
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

define spir_kernel void @test_vector_16(<16 x i2> %a2, <16 x i4> %a4, ptr addrspace(1) %res) #0 {
entry:
  ; CHECK-NATIVE-LABEL: Begin function test_vector_16
  ; CHECK-EMULATION-LABEL: Begin function test_vector_16
  ; CHECK-EMULATION-NOT: OpBitReverse

  ; CHECK-NATIVE-DAG: %[[#V16I2_Arg:]] = OpFunctionParameter %[[#V16I2]]
  ; CHECK-NATIVE-DAG: %[[#V16I4_Arg:]] = OpFunctionParameter %[[#V16I4]]

  ; CHECK-EMULATION-DAG: %[[#V16I2_Arg:]] = OpFunctionParameter %[[#V16I2]]
  ; CHECK-EMULATION-DAG: %[[#V16I4_Arg:]] = OpFunctionParameter %[[#V16I4]]

  ; CHECK-NATIVE: OpBitReverse  %[[#V16I2]] %[[#V16I2_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V16I2]] %[[#V16I2_Arg]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V16I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V16I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V16I2:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V16I2:]] %[[#]] %[[#]]
  %call2 = call <16 x i2> @llvm.bitreverse.v16i2(<16 x i2> %a2)
  store <16 x i2> %call2, ptr addrspace(1) %res, align 4

  ; CHECK-NATIVE: OpBitReverse  %[[#V16I4]] %[[#V16I4_Arg]]
  ; CHECK-EMULATION: OpShiftRightLogical %[[#V16I4]] %[[#V16I4_Arg]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V16I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseAnd %[[#V16I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpShiftLeftLogical %[[#V16I4:]] %[[#]] %[[#]]
  ; CHECK-EMULATION: OpBitwiseOr %[[#V16I4:]] %[[#]] %[[#]]
  %call4 = call <16 x i4> @llvm.bitreverse.v16i4(<16 x i4> %a4)
  store <16 x i4> %call4, ptr addrspace(1) %res, align 8

  ; CHECK-NATIVE-LABEL: OpFunctionEnd
  ; CHECK-EMULATION-LABEL: OpFunctionEnd
  ret void
}
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Declarations
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Scalar
declare i2 @llvm.bitreverse.i2(i2)
declare i4 @llvm.bitreverse.i4(i4)

; 2-element vectors
declare <2 x i2> @llvm.bitreverse.v2i2(<2 x i2>)
declare <2 x i4> @llvm.bitreverse.v2i4(<2 x i4>)

; 4-element vector declarations
declare <4 x i2> @llvm.bitreverse.v4i2(<4 x i2>)
declare <4 x i4> @llvm.bitreverse.v4i4(<4 x i4>)

; 8-element vector declarations
declare <8 x i2> @llvm.bitreverse.v8i2(<8 x i2>)
declare <8 x i4> @llvm.bitreverse.v8i4(<8 x i4>)

; 16-element vector declarations
declare <16 x i2> @llvm.bitreverse.v16i2(<16 x i2>)
declare <16 x i4> @llvm.bitreverse.v16i4(<16 x i4>)
