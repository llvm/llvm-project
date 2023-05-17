; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: %[[#short:]] = OpTypeInt 16
; CHECK-SPIRV: %[[#short2:]] = OpTypeVector %[[#short]] 2
; CHECK-SPIRV: OpBitReverse %[[#short2]]

define spir_kernel void @testBitRev(<2 x i16> %a, <2 x i16> %b, <2 x i16> %c, <2 x i16> addrspace(1)* nocapture %res) local_unnamed_addr {
entry:
  %call = tail call <2 x i16> @llvm.bitreverse.v2i16(<2 x i16> %b)
  store <2 x i16> %call, <2 x i16> addrspace(1)* %res, align 4
  ret void
}

declare <2 x i16> @llvm.bitreverse.v2i16(<2 x i16>)
