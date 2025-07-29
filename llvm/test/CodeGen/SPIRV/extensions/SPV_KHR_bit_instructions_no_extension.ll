; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-vulkan %s -o - | FileCheck %s --check-prefix=CHECK-NO-EXTENSION

; CHECK-NO-EXTENSION:     OpCapability Shader
; CHECK-NO-EXTENSION-NOT: OpCabilitity BitInstructions
; CHECK-NO-EXTENSION-NOT: OpExtension "SPV_KHR_bit_instructions"
; CHECK-NO-EXTENSION: %[[#int:]] = OpTypeInt 32
; CHECK-NO-EXTENSION: OpBitReverse %[[#int]]

define spir_kernel void @testBitRev(i32 %a, i32 %b, i32 %c, i32 addrspace(1)* nocapture %res) local_unnamed_addr #0 {
entry:
  %call = tail call i32 @llvm.bitreverse.i32(i32 %b)
  store i32 %call, i32 addrspace(1)* %res, align 4
  ret void
}

declare i32 @llvm.bitreverse.i32(i32)
attributes #0 = { "hlsl.shader"="compute" }
