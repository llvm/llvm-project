; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CL --implicit-check-not=FPFastMathMode
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK,CL --implicit-check-not=FPFastMathMode
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s --check-prefixes=CHECK,VK --implicit-check-not=FPFastMathMode
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CL-DAG: %[[#ExtInstId:]] = OpExtInstImport "OpenCL.std"
; VK-DAG: %[[#ExtInstId:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Double:]] = OpTypeFloat 64
; CHECK-DAG: %[[#Double2:]] = OpTypeVector %[[#Double]] 2
; CL-DAG: %[[#Zero:]] = OpConstantNull %[[#Float]]
; VK-DAG: %[[#Zero:]] = OpConstant %[[#Float]] 0{{$}}
; CHECK-DAG: %[[#One:]] = OpConstant %[[#Float]] 1{{$}}

; CHECK: %[[#X:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#Y:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#X1:]] = OpSelect %[[#Float]] %[[#]] %[[#Y]] %[[#X]]
; CHECK: %[[#Y1:]] = OpSelect %[[#Float]] %[[#]] %[[#X1]] %[[#Y]]
; CL: %[[#Min:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] fmin %[[#X1]] %[[#Y1]]
; CL: %[[#XSign:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] copysign %[[#One]] %[[#X1]]
; VK: %[[#Min:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] NMin %[[#X1]] %[[#Y1]]
; VK: %[[#XSign:]] = OpBitcast %[[#Float]]
; CHECK: %[[#XNeg:]] = OpFOrdLessThan %[[#]] %[[#XSign]] %[[#Zero]]
; CHECK: %[[#S:]] = OpSelect %[[#Float]] %[[#XNeg]] %[[#X1]] %[[#Y1]]
; CL: %[[#R:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] copysign %[[#Min]] %[[#S]]
; VK: %[[#R:]] = OpBitcast %[[#Float]]
; CHECK: OpReturnValue %[[#R]]
define spir_func float @test_minimumnum(float %x, float %y) {
  %r = call float @llvm.minimumnum.f32(float %x, float %y)
  ret float %r
}

; CHECK: OpFunctionParameter %[[#Double2]]
; CL: %[[#Max:]] = OpExtInst %[[#Double2]] %[[#ExtInstId]] fmax %[[#Xv:]] %[[#Yv:]]
; VK: %[[#Max:]] = OpExtInst %[[#Double2]] %[[#ExtInstId]] NMax %[[#Xv:]] %[[#Yv:]]
; CHECK: %[[#XNegv:]] = OpFOrdLessThan
; CHECK: %[[#Sv:]] = OpSelect %[[#Double2]] %[[#XNegv]] %[[#Yv]] %[[#Xv]]
; CL: %[[#Rv:]] = OpExtInst %[[#Double2]] %[[#ExtInstId]] copysign %[[#Max]] %[[#Sv]]
; VK: %[[#Rv:]] = OpBitcast %[[#Double2]]
; CHECK: OpReturnValue %[[#Rv]]
define spir_func <2 x double> @test_maximumnum_v2(<2 x double> %x, <2 x double> %y) {
  %r = call <2 x double> @llvm.maximumnum.v2f64(<2 x double> %x, <2 x double> %y)
  ret <2 x double> %r
}

; CHECK: %[[#Xn:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#Yn:]] = OpFunctionParameter %[[#Float]]
; CHECK-NEXT: OpLabel
; CL-NEXT: %[[#Rn:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] fmin %[[#Xn]] %[[#Yn]]
; VK-NEXT: %[[#Rn:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] NMin %[[#Xn]] %[[#Yn]]
; CHECK-NEXT: OpReturnValue %[[#Rn]]
define spir_func float @test_minimumnum_nnan_nsz_reassoc(float %x, float %y) {
  %r = call nnan nsz reassoc float @llvm.minimumnum.f32(float %x, float %y)
  ret float %r
}

;; A signaling NaN operand yields the other operand.
; CHECK: OpFunction %[[#Float]] None
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpReturnValue %[[#One]]
define spir_func float @test_minimumnum_snan() {
  %r = call nsz float @llvm.minimumnum.f32(float 0x7FF0000020000000, float 1.0)
  ret float %r
}

;; nnan alone drops the NaN selects but keeps the sign fix.
; CHECK-LABEL: Begin function test_nnan_only
; CHECK: OpFunctionParameter %[[#Float]]
; CHECK: OpFunctionParameter %[[#Float]]
; CHECK-NOT: OpUnordered
; CHECK-NOT: OpIsNan
; CL: OpExtInst %[[#Float]] %[[#ExtInstId]] fmin
; VK: OpExtInst %[[#Float]] %[[#ExtInstId]] NMin
; CHECK: OpFOrdLessThan
; CHECK: OpReturnValue
define spir_func float @test_nnan_only(float %x, float %y) {
  %r = call nnan float @llvm.minimumnum.f32(float %x, float %y)
  ret float %r
}

;; nsz alone keeps the NaN selects but drops the sign fix.
; CHECK-LABEL: Begin function test_nsz_only
; CHECK: OpFunctionParameter %[[#Float]]
; CHECK: OpFunctionParameter %[[#Float]]
; CL: OpUnordered
; VK: OpIsNan
; CHECK: OpSelect
; CHECK: OpSelect
; CL: %[[#Rz:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] fmin
; VK: %[[#Rz:]] = OpExtInst %[[#Float]] %[[#ExtInstId]] NMin
; CHECK-NEXT: OpReturnValue %[[#Rz]]
define spir_func float @test_nsz_only(float %x, float %y) {
  %r = call nsz float @llvm.minimumnum.f32(float %x, float %y)
  ret float %r
}
