; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: %[[#HALF:]] = OpTypeFloat 16
; CHECK-SPIRV: %[[#HALFPTR:]] = OpTypePointer Function %[[#HALF]]
; CHECK-SPIRV: %[[#HALFV2:]] = OpTypeVector %[[#HALF]] 2
; CHECK-SPIRV: %[[#HALFV2PTR:]] = OpTypePointer Function %[[#HALFV2]]
; CHECK-SPIRV: %[[#CONST:]] = OpConstant %[[#HALF]] 14788
; CHECK-SPIRV: %[[#ADDR:]] = OpVariable %[[#HALFPTR]] Function
; CHECK-SPIRV: %[[#ADDR2:]] = OpVariable %[[#HALFV2PTR]] Function
; CHECK-SPIRV: %[[#]] = OpExtInst %[[#HALF]] %[[#]] fract %[[#CONST]] %[[#ADDR]]
; CHECK-SPIRV: %[[#]] = OpExtInst %[[#HALFV2]] %[[#]] fract %[[#]] %[[#ADDR2]]

define spir_kernel void @test() {
entry:
  %addr = alloca half
  %addr2 = alloca <2 x half>
  %res = call spir_func noundef half @_Z17__spirv_ocl_fractDF16_PU3AS0DF16_(half noundef 0xH39C4, ptr noundef %addr)
  %res2 = call spir_func noundef <2 x half> @_Z17__spirv_ocl_fractDv2_DF16_PU3AS0S_(<2 x half> noundef <half 0xH39C4, half 0xH0000>, ptr noundef %addr2)
  ret void
}

declare spir_func noundef half @_Z17__spirv_ocl_fractDF16_PU3AS0DF16_(half noundef, ptr noundef) local_unnamed_addr

declare spir_func noundef <2 x half> @_Z17__spirv_ocl_fractDv2_DF16_PU3AS0S_(<2 x half> noundef, ptr noundef) local_unnamed_addr
