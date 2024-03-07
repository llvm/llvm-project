; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpSelect

;; LLVM IR was generated with -cl-std=c++ option

define spir_kernel void @test(i32 %op1, i32 %op2) {
entry:
  %0 = trunc i8 undef to i1
  %call = call spir_func i32 @_Z14__spirv_Selectbii(i1 zeroext %0, i32 %op1, i32 %op2)
  ret void
}

declare spir_func i32 @_Z14__spirv_Selectbii(i1 zeroext, i32, i32)
