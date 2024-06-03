; The goal of the test case is to ensure that there's no crash
; on translation of integers with bit width less than 8.

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: OpCapability BitInstructions
; CHECK-SPIRV: OpExtension "SPV_KHR_bit_instructions"
; CHECK-SPIRV: %[[#CharTy:]] = OpTypeInt 8 0
; CHECK-SPIRV: %[[#]] = OpBitReverse %[[#CharTy]] %[[#]]

define spir_func signext i2 @foo(i2 noundef signext %a) {
entry:
  %b = tail call i2 @llvm.bitreverse.i2(i2 %a)
  ret i2 %b
}

declare i2 @llvm.bitreverse.i2(i2)
