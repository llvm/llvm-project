; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_tensor_float32_conversion %s -o - %.spvasm

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @test(<8 x i16> %in) {
  %res = tail call spir_func <8 x i16> @_Z27__spirv_RoundFToTF32INTELDv8_s(<8 x i16> %in)
  ret void
}

declare spir_func float @_Z27__spirv_RoundFToTF32INTELDv8_s(<8 x i16>)
