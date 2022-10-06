; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV:     %[[#TypeImage:]] = OpTypeImage
; CHECK-SPIRV-NOT: OpTypeImage
; CHECK-SPIRV:     %[[#]] = OpTypeFunction %[[#]] %[[#TypeImage]]
; CHECK-SPIRV:     %[[#]] = OpTypeFunction %[[#]] %[[#TypeImage]]
; CHECK-SPIRV:     %[[#]] = OpFunctionParameter %[[#TypeImage]]
; CHECK-SPIRV:     %[[#]] = OpFunctionParameter %[[#TypeImage]]
; CHECK-SPIRV:     %[[#ParamID:]] = OpFunctionParameter %[[#TypeImage]]
; CHECK-SPIRV:     %[[#]] = OpFunctionCall %[[#]] %[[#]] %[[#ParamID]]

%opencl.image2d_ro_t = type opaque

define spir_func void @f0(%opencl.image2d_ro_t addrspace(1)* %v2, <2 x float> %v3) {
entry:
  ret void
}

define spir_func void @f1(%opencl.image2d_ro_t addrspace(1)* %v2, <2 x float> %v3) {
entry:
  ret void
}

define spir_kernel void @test(%opencl.image2d_ro_t addrspace(1)* %v1) {
entry:
  call spir_func void @f0(%opencl.image2d_ro_t addrspace(1)* %v1, <2 x float> <float 1.000000e+00, float 5.000000e+00>)
  ret void
}
