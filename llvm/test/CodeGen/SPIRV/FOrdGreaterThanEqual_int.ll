; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: %[[#result:]] = OpFOrdGreaterThanEqual %[[#]] %[[#]] %[[#]]
; CHECK-SPIRV: %[[#]] = OpSelect %[[#]] %[[#result]] %[[#]] %[[#]]

;; LLVM IR was generated with -cl-std=c++ option

define spir_kernel void @test(float %op1, float %op2) {
entry:
  %call = call spir_func i32 @_Z14isgreaterequalff(float %op1, float %op2)
  ret void
}

declare spir_func i32 @_Z14isgreaterequalff(float, float)
