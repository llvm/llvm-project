; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - | FileCheck %s --check-prefix=CHECK-EXTENSION
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.2-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - -filetype=obj | spirv-val --target-env opencl2.2 %} 
;
; CHECK-EXTENSION: Capability BitInstructions
; CHECK-EXTENSION: Extension "SPV_KHR_bit_instructions"
;
; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] None %[[#]]
; CHECK-EXTENSION: %[[#reversebase:]] = OpFunctionParameter %[[#]]
; CHECK-EXTENSION: %[[#]] = OpBitReverse %[[#]] %[[#reversebase]]
; OpenCL equivalent.
; kernel void testBitReverse_SPIRVFriendly(long4 b, global long4 *res) {
;   *res = bit_reverse(b);
; }
define spir_kernel void @testBitReverse_SPIRVFriendly(<4 x i64> %b, ptr addrspace(1) %res) {
entry:
  %call = call <4 x i64> @llvm.bitreverse.v4i64(<4 x i64> %b)
  store <4 x i64> %call, ptr addrspace(1) %res
  ret void
}

declare <4 x i64> @llvm.bitreverse.v4i64(<4 x i64>)
