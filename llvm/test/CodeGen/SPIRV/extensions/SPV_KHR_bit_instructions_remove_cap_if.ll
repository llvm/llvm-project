; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - | FileCheck %s

; CHECK: OpCapability Shader
; CHECK-NOT: OpCapability BitInstructions
; CHECK: OpExtension "SPV_KHR_bit_instructions"
; CHECK: OpBitReverse

define spir_func i32 @testBitRev(i32 %a) {
entry:
  %call = call i32 @llvm.bitreverse.i32(i32 %a)
  ret i32 %call
}

define void @main() #0 {
  ret void
}

declare i32 @llvm.bitreverse.i32(i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
