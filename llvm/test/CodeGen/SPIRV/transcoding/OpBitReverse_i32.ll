; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: %[[#int:]] = OpTypeInt 32
; CHECK-SPIRV: OpBitReverse %[[#int]]

define spir_kernel void @testBitRev(i32 %a, i32 %b, i32 %c, i32 addrspace(1)* nocapture %res) local_unnamed_addr {
entry:
  %call = tail call i32 @llvm.bitreverse.i32(i32 %b)
  store i32 %call, i32 addrspace(1)* %res, align 4
  ret void
}

declare i32 @llvm.bitreverse.i32(i32)
