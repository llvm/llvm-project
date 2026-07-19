; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - | FileCheck %s --check-prefix=CHECK-EXTENSION
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.2-unknown-unknown %s --spirv-ext=+SPV_KHR_bit_instructions -o - -filetype=obj | spirv-val --target-env opencl2.2 %}

; CHECK-EXTENSION:      OpCapability BitInstructions
; CHECK-EXTENSION-NEXT: OpExtension "SPV_KHR_bit_instructions"
; CHECK-EXTENSION-NOT:  OpCabilitity Shader
; CHECK-EXTENSION: %[[#int:]] = OpTypeInt 32
; CHECK-EXTENSION: OpBitReverse %[[#int]]

define spir_kernel void @testBitRev(i32 %a, i32 %b, i32 %c, ptr addrspace(1) nocapture %res) local_unnamed_addr {
entry:
  %call = tail call i32 @llvm.bitreverse.i32(i32 %b)
  store i32 %call, ptr addrspace(1) %res, align 4
  ret void
}

declare i32 @llvm.bitreverse.i32(i32)
