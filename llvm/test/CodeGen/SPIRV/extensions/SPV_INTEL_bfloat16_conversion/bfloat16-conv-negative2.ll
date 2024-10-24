; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_INTEL_bfloat16_conversion %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: result and argument must have the same number of components

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

define spir_func void @test(<8 x float> %in) {
  %res = tail call spir_func <4 x i16> @_Z27__spirv_ConvertFToBF16INTELDv8_f(<8 x float> %in)
  ret void
}

declare spir_func <4 x i16> @_Z27__spirv_ConvertFToBF16INTELDv8_f(<8 x float>)
