; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV:      OpBitwiseAnd
; CHECK-SPIRV-NEXT: OpINotEqual

define spir_kernel void @test(i32 %op1, i32 %op2, i8 %op3, ptr addrspace(1) %out) {
entry:
  %0 = trunc i8 %op3 to i1
  %call = call spir_func i32 @_Z14__spirv_Selectbii(i1 zeroext %0, i32 %op1, i32 %op2)
  store i32 %call, ptr addrspace(1) %out
  ret void
}

declare spir_func i32 @_Z14__spirv_Selectbii(i1 zeroext, i32, i32)
