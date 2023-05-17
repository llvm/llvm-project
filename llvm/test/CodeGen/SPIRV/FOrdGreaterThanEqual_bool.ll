; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV:     OpFOrdGreaterThanEqual
; CHECK-SPIRV-NOT: OpSelect

;; LLVM IR was generated with -cl-std=c++ option

define spir_kernel void @test(float %op1, float %op2) {
entry:
  %0 = call spir_func zeroext i1 @_Z28__spirv_FOrdGreaterThanEqualff(float %op1, float %op2)
  ret void
}

declare spir_func zeroext i1 @_Z28__spirv_FOrdGreaterThanEqualff(float, float)
