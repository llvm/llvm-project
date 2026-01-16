; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check OpenCL built-in shuffle and shuffle2 translation.

; CHECK-SPIRV: %[[#]] = OpExtInst %[[#]] %[[#]] shuffle %[[#]] %[[#]]
; CHECK-SPIRV: %[[#]] = OpExtInst %[[#]] %[[#]] shuffle2 %[[#]] %[[#]] %[[#]]

define spir_kernel void @test() {
entry:
  %call = call spir_func <2 x float> @_Z7shuffleDv2_fDv2_j(<2 x float> zeroinitializer, <2 x i32> zeroinitializer)
  ret void
}

declare spir_func <2 x float> @_Z7shuffleDv2_fDv2_j(<2 x float>, <2 x i32>)

define spir_kernel void @test2() {
entry:
  %call = call spir_func <4 x float> @_Z8shuffle2Dv2_fS_Dv4_j(<2 x float> zeroinitializer, <2 x float> zeroinitializer, <4 x i32> zeroinitializer)
  ret void
}

declare spir_func <4 x float> @_Z8shuffle2Dv2_fS_Dv4_j(<2 x float>, <2 x float>, <4 x i32>)
