; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-NO-EXTENSION
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %} 
;
; CHECK-NO-EXTENSION-NOT: Capability BitInstructions 
; CHECK-NO-EXTENSION-NOT: Extension "SPV_KHR_bit_instructions"
; CHECK-NO-EXTENSION: Capability Shader 

define internal spir_func void @testBitReverse_SPIRVFriendly() #3 {
entry:
  %call = call <4 x i64> @llvm.bitreverse.v4i64(<4 x i64> <i64 1, i64 2, i64 3, i64 4>)
  ret void
}

declare <4 x i64> @llvm.bitreverse.v4i64(<4 x i64>)

attributes #3 = { nounwind "hlsl.shader"="compute" }
