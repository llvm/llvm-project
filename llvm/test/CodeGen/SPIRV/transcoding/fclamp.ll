; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV:     %[[#]] = OpExtInst %[[#]] %[[#]] fclamp
; CHECK-SPIRV-NOT: %[[#]] = OpExtInst %[[#]] %[[#]] clamp

define spir_kernel void @test_scalar(float addrspace(1)* nocapture readonly %f) {
entry:
  %0 = load float, float addrspace(1)* %f, align 4
  %call = tail call spir_func float @_Z5clampfff(float %0, float 0.000000e+00, float 1.000000e+00)
  %1 = load float, float addrspace(1)* %f, align 4
  %conv = fptrunc float %1 to half
  %call1 = tail call spir_func half @_Z5clampDhDhDh(half %conv, half %conv, half %conv)
  ret void
}

declare spir_func float @_Z5clampfff(float, float, float)

declare spir_func half @_Z5clampDhDhDh(half, half, half)
