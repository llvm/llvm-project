; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: OpSelect

;; LLVM IR was generated with -cl-std=c++ option

define spir_kernel void @test(i32 %op1, i32 %op2) {
entry:
  %truncated = trunc i8 undef to i1
  %call = call spir_func i32 @_Z14__spirv_Selectbii(i1 zeroext %truncated, i32 %op1, i32 %op2)
  ret void
}

declare spir_func i32 @_Z14__spirv_Selectbii(i1 zeroext, i32, i32)
